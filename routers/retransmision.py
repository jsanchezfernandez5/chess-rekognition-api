"""
Router de retransmisión de partidas de ajedrez en tiempo real.

Endpoints:
    POST /retransmision/host                | Crea una nueva retransmisión con token único y la marca como activa.
    GET  /retransmision/status/{token}      | Devuelve si la retransmisión está activa y cuántos viewers hay conectados.
    PATCH /retransmision/{id}               | Actualiza los metadatos de una retransmisión del usuario autenticado.
    WS   /retransmision/ws/host/{token}     | WebSocket del emisor (host): recibe el estado del tablero (y opcionalmente frames de vídeo) y hace broadcast a los viewers.
    WS   /retransmision/ws/viewer/{token}   | WebSocket del espectador (viewer): recibe actualizaciones en tiempo real del tablero (y vídeo si el host lo comparte).

Relay de vídeo OPCIONAL: el host puede enviar mensajes JSON {"type": "video_frame", "frame":
"<jpeg base64>"} por su mismo WebSocket. El servidor actúa SOLO como relé (no decodifica ni
procesa imagen): reenvía el frame a todos los viewers SIN cachearlo — la caché de estado
(self.states) sigue reservada al último FEN/PGN para los late-joiners.
"""
import asyncio
import json
import random
import string
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database import get_db
from core.dependencies import get_current_user
from core.models import Retransmision, RetransmisionCreate, RetransmisionResponse, Usuario

# Creamos el router para las retransmisiones.
router = APIRouter(prefix="/retransmision", tags=["Retransmisión"], responses={404: {"description": "No encontrado"}},)

# Límite de tamaño (en caracteres del texto JSON) por frame de vídeo retransmitido.
# Con JPEG 480px calidad ~0.5 los frames rondan 15-30 KB; el margen cubre picos y
# descarta abusos sin romper el relay normal.
MAX_VIDEO_FRAME_BYTES = 200_000

# Función interna para generar tokens únicos.
def _generate_token(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))

# -------------------------------------------------------------------------------
# SCHEMA DE REQUEST
# Define los datos que el cliente debe enviar en el body del POST /retransmision/host y PATCH /retransmision/{id}.
# Pydantic valida automáticamente los tipos y restricciones antes de ejecutar el endpoint.
# -------------------------------------------------------------------------------
class RetransmisionStatus(BaseModel):
    token: str
    active: bool
    viewers: int

# -------------------------------------------------------------------------------
# CONNECTION MANAGER
# Clase que centraliza toda la lógica de conexiones WebSocket.
# Mantiene en memoria tres diccionarios indexados por token:
#   - hosts:   un solo WebSocket por token (el emisor de la partida)
#   - viewers: lista de WebSockets por token (los espectadores)
#   - states:  último estado del tablero recibido del host (para enviarlo a viewers que se conecten tarde)
# -------------------------------------------------------------------------------
class ConnectionManager:
    """
    Gestiona las conexiones WebSocket de emisores (hosts) y espectadores (viewers).
    """
    # Conexiones WebSocket de los viewers.
    def __init__(self):
        # Dict[token → List[WebSocket]]: viewers conectados por retransmisión
        self.viewers: Dict[str, List[WebSocket]] = {}

        # Dict[token → dict]: último estado del tablero enviado por el host para cada retransmisión
        self.states:  Dict[str, dict] = {}

        # Dict[token → WebSocket]: conexión del host por retransmisión (solo uno por token)
        self.hosts:   Dict[str, WebSocket] = {}

    # Método para conectar un viewer a la retransmisión.
    async def connect_viewer(self, websocket: WebSocket, token: str):
        """Acepta la conexión del viewer y le envía el último estado conocido del tablero."""
        await websocket.accept()
        self.viewers.setdefault(token, []).append(websocket)
        if token in self.states:
            await websocket.send_json(self.states[token])

    # Método para desconectar un viewer de la retransmisión.
    def disconnect_viewer(self, websocket: WebSocket, token: str):
        """Elimina el viewer de la lista de conexiones activas."""
        if token in self.viewers and websocket in self.viewers[token]:
            self.viewers[token].remove(websocket)

    # Método para conectar un host a la retransmisión.
    async def connect_host(self, websocket: WebSocket, token: str):
        """Acepta la conexión del host y registra su WebSocket."""
        await websocket.accept()
        self.hosts[token] = websocket
        self.viewers.setdefault(token, [])

    # Método para desconectar un host de la retransmisión.
    async def disconnect_host(self, token: str, db: Session):
        """
        Desconecta el host, marca la retransmisión como inactiva en la base de datos y cierra todas las conexiones de viewers con código 1000 (cierre normal).
        """
        # Elimina el host.
        self.hosts.pop(token, None)
        
        # Cierra la retransmisión en la base de datos.
        if db:
            r = db.query(Retransmision).filter(Retransmision.token == token).first()
            if r:
                r.is_activa = False
                db.commit()

        # Elimina el estado y los viewers de la retransmisión.
        self.states.pop(token, None)
        for viewer in self.viewers.pop(token, []):
            try:
                await viewer.close(code=1000, reason="Host finalizó la retransmisión")
            except Exception:
                pass

    # Método para hacer broadcast a los viewers de una retransmisión.
    async def broadcast_to_viewers(self, token: str, message: dict):
        """
        Guarda el estado del tablero y lo envía a todos los viewers conectados.
        Si un viewer falla al recibir, se desconecta limpiamente.
        """
        self.states[token] = message
        dead = []
        for viewer in self.viewers.get(token, []):
            try:
                await viewer.send_json(message)
            except Exception:
                dead.append(viewer)

        # Elimina los viewers con conexión rota fuera del bucle para no modificar la lista mientras se itera
        for d in dead:
            self.disconnect_viewer(d, token)

    # Método para retransmitir un frame de vídeo a los viewers SIN cachearlo.
    async def relay_video_frame(self, token: str, message: dict):
        """
        Relé puro de frames de vídeo del host hacia todos los viewers conectados.

        A diferencia de broadcast_to_viewers(), NO toca self.states: la caché de estado debe
        seguir conteniendo el último FEN/PGN (es lo que reciben los late-joiners), nunca un
        frame de vídeo obsoleto. Los viewers que llegan tarde simplemente empiezan a ver
        vídeo en el siguiente frame.
        """
        dead = []
        for viewer in self.viewers.get(token, []):
            try:
                await viewer.send_json(message)
            except Exception:
                dead.append(viewer)

        for d in dead:
            self.disconnect_viewer(d, token)

# Instancia Singleton del gestor de conexiones.
manager = ConnectionManager()

# -------------------------------------------------------------------------------
# [ENDPOINT] - POST /retransmision/host
# Crea una nueva retransmisión con token único y la marca como activa.
# El token se genera aleatoriamente y se comprueba que no exista en la base de datos.
# -------------------------------------------------------------------------------
@router.post(
    "/host", 
    response_model=RetransmisionResponse,
    summary="Crea una nueva retransmisión con token único y la marca como activa."
)
def init_retransmision(
    datos: RetransmisionCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
):
    """
    Crea una nueva retransmisión con token único y la marca como activa.

    El token se genera aleatoriamente y se comprueba que no exista en la base de datos.
    """
    token = _generate_token()

    # Genera un token único, comprobando que no exista en la base de datos.
    while db.query(Retransmision).filter(Retransmision.token == token).first():
        token = _generate_token()

    # Crea la retransmisión en la base de datos.
    r = Retransmision(token=token, username=current_user.username, is_activa=True, **datos.model_dump())
    db.add(r)
    db.commit()
    db.refresh(r)

    # Devuelve la retransmisión creada.
    return r

# -------------------------------------------------------------------------------
# [ENDPOINT] - GET /retransmision/status/{token}
# Devuelve si la retransmisión está activa y cuántos viewers hay conectados.
# -------------------------------------------------------------------------------
@router.get(
    "/status/{token}", 
    summary="Devuelve si la retransmisión está activa y cuántos viewers hay conectados."
)
async def get_retransmision_status(token: str):
    """
    Devuelve si la retransmisión está activa y cuántos viewers hay conectados.
    """
    return {
        "success": True,
        "data": {
            "token":   token,
            "active":  token in manager.hosts or token in manager.states,
            "viewers": len(manager.viewers.get(token, [])),
        },
    }

# -------------------------------------------------------------------------------
# [ENDPOINT] - PATCH /retransmision/{id}
# Actualiza los metadatos de una retransmisión del usuario autenticado.
# -------------------------------------------------------------------------------
@router.patch(
    "/{id_retransmision}", 
    summary="Actualiza los metadatos de una retransmisión del usuario autenticado."
)
def update_retransmision(
    id_retransmision: int,
    datos: RetransmisionCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
):
    """
    Actualiza los metadatos de una retransmisión del usuario autenticado.
    """
    # Busca la retransmisión por ID y usuario autenticado.
    r = db.query(Retransmision).filter(
        Retransmision.id_retransmision == id_retransmision,
        Retransmision.username == current_user.username,
    ).first()

    # Lanza excepción si no se encuentra la retransmisión.
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Retransmisión no encontrada o sin permiso.")
    for key, value in datos.model_dump(exclude_unset=True).items():
        setattr(r, key, value)

    # Guarda los cambios y devuelve la retransmisión actualizada.
    db.commit()
    db.refresh(r)
    return r

# -------------------------------------------------------------------------------
# [ENDPOINT] - WS /retransmision/ws/host/{token}
# WebSocket del emisor: recibe estado del tablero y hace broadcast a los viewers.
# -------------------------------------------------------------------------------
@router.websocket(
    "/ws/host/{token}"
)
async def websocket_host(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """
    WebSocket del emisor: recibe estado del tablero y hace broadcast a los viewers.
    """
    # Conecta el host al WebSocket.
    await manager.connect_host(websocket, token)
    try:
        # Bucle infinito para recibir actualizaciones del host.
        while True:
            try:
                data_text = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.close(code=1001, reason="Heartbeat timeout")
                except Exception:
                    pass
                await manager.disconnect_host(token, db)
                break

            if data_text == "ping":
                await websocket.send_text("pong")
                continue

            try:
                data = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            # Relay de vídeo OPCIONAL: se reenvía sin cachear (ver relay_video_frame).
            # Se limita el tamaño por frame para que un host desbocado no sature el ancho de banda.
            if isinstance(data, dict) and data.get("type") == "video_frame":
                if len(data_text) <= MAX_VIDEO_FRAME_BYTES:
                    await manager.relay_video_frame(token, data)
                continue

            await manager.broadcast_to_viewers(token, data)
    except WebSocketDisconnect:
        await manager.disconnect_host(token, db)

# -------------------------------------------------------------------------------
# [ENDPOINT] - WS /retransmision/ws/viewer/{token}
# WebSocket del espectador: recibe actualizaciones en tiempo real del tablero.
# -------------------------------------------------------------------------------
@router.websocket(
    "/ws/viewer/{token}"
)
async def websocket_viewer(websocket: WebSocket, token: str):
    """
    WebSocket del espectador: recibe actualizaciones en tiempo real del tablero.
    """
    # Conecta el viewer al WebSocket.
    await manager.connect_viewer(websocket, token)
    try:
        # Bucle infinito para recibir actualizaciones del viewer.
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect_viewer(websocket, token)

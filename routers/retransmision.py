"""
Endpoints HTTP y WebSocket para la retransmisión de partidas en tiempo real.
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
router = APIRouter(
    prefix="/retransmision",
    tags=["Retransmisión en tiempo real"],
    responses={404: {"description": "No encontrado"}},
)

# Función para generar tokens únicos.
def _generate_token(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))

# Modelo para obtener el estado de una retransmisión.
class RetransmisionStatus(BaseModel):
    token: str
    active: bool
    viewers: int

# Clase para gestionar las conexiones WebSocket de emisores (hosts) y espectadores (viewers).
class ConnectionManager:
    """
    Gestiona las conexiones WebSocket de emisores (hosts) y espectadores (viewers).
    """
    # Conexiones WebSocket de los viewers.
    def __init__(self):
        self.viewers: Dict[str, List[WebSocket]] = {}
        self.states:  Dict[str, dict] = {}
        self.hosts:   Dict[str, WebSocket] = {}

    # Método para conectar un viewer a la retransmisión.
    async def connect_viewer(self, websocket: WebSocket, token: str):
        await websocket.accept()
        self.viewers.setdefault(token, []).append(websocket)
        if token in self.states:
            await websocket.send_json(self.states[token])

    # Método para desconectar un viewer de la retransmisión.
    def disconnect_viewer(self, websocket: WebSocket, token: str):
        if token in self.viewers and websocket in self.viewers[token]:
            self.viewers[token].remove(websocket)

    # Método para conectar un host a la retransmisión.
    async def connect_host(self, websocket: WebSocket, token: str):
        await websocket.accept()
        self.hosts[token] = websocket
        self.viewers.setdefault(token, [])

    # Método para desconectar un host de la retransmisión.
    async def disconnect_host(self, token: str, db: Session):
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

    # Método para enviar un mensaje a todos los viewers de una retransmisión.
    async def broadcast_to_viewers(self, token: str, message: dict):
        self.states[token] = message
        dead = []
        for viewer in self.viewers.get(token, []):
            try:
                await viewer.send_json(message)
            except Exception:
                dead.append(viewer)
        for d in dead:
            self.disconnect_viewer(d, token)

# Instancia del gestor de conexiones.
manager = ConnectionManager()

# Endpoint para inicializar una nueva retransmisión.
@router.post(
    "/host", 
    response_model=RetransmisionResponse,
     summary="Inicializar una nueva retransmisión"
)
def init_retransmision(
    datos: RetransmisionCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
):
    """
    Crea una retransmisión con token único y la marca como activa.
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

# Endpoint para obtener el estado de una retransmisión por token.
@router.get(
    "/status/{token}", 
    summary="Obtener el estado de una retransmisión por token"
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

# Endpoint para actualizar una retransmisión.
@router.patch(
    "/{id_retransmision}", 
    summary="Actualizar retransmisión"
)
def update_retransmision(
    id_retransmision: int,
    datos: RetransmisionCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user),
):
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

# Endpoint WebSocket para conectar el host al WebSocket.
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

            await manager.broadcast_to_viewers(token, data)
    except WebSocketDisconnect:
        await manager.disconnect_host(token, db)

# Endpoint WebSocket para conectar el viewer al WebSocket.
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

"""Módulo de retransmisión de partidas en tiempo real.

Proporciona endpoints HTTP y WebSocket para gestionar
retransmisiones de partidas de ajedrez, permitiendo que
un emisor (host) difunda el estado del tablero a múltiples
espectadores (viewers) en tiempo real.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from typing import Dict, List
from pydantic import BaseModel
from sqlalchemy.orm import Session
import string
import random

from db.database import get_db
from models.usuarios import Usuario
from core.dependencies import get_current_user
from schemas.retransmisiones import RetransmisionCreate, RetransmisionResponse
from models.retransmisiones import Retransmision

router = APIRouter(
    prefix="/retransmision",
    tags=["Retransmisión en tiempo real"],
    responses={404: {"description": "No encontrado"}},
)

def generate_token(length=8):
    """Genera un token alfanumérico único para identificar una retransmisión.

    Crea una cadena aleatoria de la longitud especificada usando
    letras minúsculas y dígitos.

    Args:
        length: Longitud del token a generar (por defecto 8).

    Returns:
        str con el token generado.
    """
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

class RetransmisionStatus(BaseModel):
    token: str
    active: bool
    viewers: int

class ConnectionManager:
    """Gestiona las conexiones WebSocket de emisores y espectadores.

    Mantiene el estado de las salas de retransmisión, incluyendo
    las conexiones de hosts y viewers, el último estado enviado
    y las operaciones de difusión de mensajes.
    """
    def __init__(self):
        # lista de websockets de espectadores
        self.viewers: Dict[str, List[WebSocket]] = {}
        # último estado enviado (para enviar a nuevos espectadores inmediatamente)
        self.states: Dict[str, dict] = {}
        # websocket del emisor
        self.hosts: Dict[str, WebSocket] = {}

    async def connect_viewer(self, websocket: WebSocket, token: str):
        """Conecta un nuevo espectador a una sala de retransmisión.

        Acepta la conexión WebSocket, registra al espectador en la
        sala identificada por el token y le envía el estado más
        reciente del tablero si existe.

        Args:
            websocket: Conexión WebSocket del espectador.
            token: Token identificador de la retransmisión.
        """
        await websocket.accept()
        if token not in self.viewers:
            self.viewers[token] = []
        self.viewers[token].append(websocket)
        # Envía el estado más reciente al espectador recién conectado
        if token in self.states:
            await websocket.send_json(self.states[token])

    def disconnect_viewer(self, websocket: WebSocket, token: str):
        """Desconecta un espectador de una sala de retransmisión.

        Elimina al espectador de la lista de conexiones activas
        de la sala identificada por el token.

        Args:
            websocket: Conexión WebSocket del espectador a desconectar.
            token: Token identificador de la retransmisión.
        """
        if token in self.viewers and websocket in self.viewers[token]:
            self.viewers[token].remove(websocket)

    async def connect_host(self, websocket: WebSocket, token: str):
        """Conecta un emisor (host) a una sala de retransmisión.

        Acepta la conexión WebSocket y registra al emisor como
        el host de la sala identificada por el token. Inicializa
        la lista de espectadores si es necesario.

        Args:
            websocket: Conexión WebSocket del host.
            token: Token identificador de la retransmisión.
        """
        await websocket.accept()
        self.hosts[token] = websocket
        if token not in self.viewers:
            self.viewers[token] = []

    async def disconnect_host(self, token: str, db):
        """Desconecta un emisor y finaliza la retransmisión.

        Elimina al host de la sala, marca la retransmisión como
        inactiva en la base de datos, limpia el estado en memoria
        y desconecta a todos los espectadores activos.

        Args:
            token: Token identificador de la retransmisión.
            db: Sesión de base de datos para persistir el cambio.
        """
        if token in self.hosts:
            del self.hosts[token]
        
        # Sincronización con la Base de Datos
        if db:
            retransmision = db.query(Retransmision).filter(Retransmision.token == token).first()
            if retransmision:
                retransmision.is_activa = False
                db.commit()

        # Limpieza de memoria (evitar fugas de salas huérfanas)
        if token in self.states:
            del self.states[token]
            
        if token in self.viewers:
            # Desconectar a todos los espectadores activos
            for viewer in self.viewers[token]:
                try:
                    await viewer.close(code=1000, reason="Host finalizó la retransmisión")
                except Exception:
                    pass
            del self.viewers[token]

    async def broadcast_to_viewers(self, token: str, message: dict):
        """Envía un mensaje a todos los espectadores de una sala.

        Almacena el mensaje como el estado más reciente y lo
        distribuye a todos los espectadores conectados. Si algún
        espectador tiene una conexión muerta, lo desconecta.

        Args:
            token: Token identificador de la retransmisión.
            message: Datos del estado del tablero a difundir.
        """
        self.states[token] = message
        if token in self.viewers:
            dead_connections = []
            for viewer in self.viewers[token]:
                try:
                    await viewer.send_json(message)
                except Exception:
                    dead_connections.append(viewer)
            
            for dead in dead_connections:
                self.disconnect_viewer(dead, token)

manager = ConnectionManager()

@router.post(
    "/host", 
    response_model=RetransmisionResponse, 
    summary="Inicializar una nueva retransmisión"
)
def init_retransmision(
    datos: RetransmisionCreate, 
    db: Session = Depends(get_db), 
    current_user: Usuario = Depends(get_current_user)
):
    """Inicializa una nueva retransmisión y genera un token único.

    Crea un registro de retransmisión en la base de datos con los
    datos proporcionados, genera un token alfanumérico único y
    marca la retransmisión como activa.

    Args:
        datos: Datos de la retransmisión (blancas, negras, evento, etc.).
        db: Sesión de base de datos.
        current_user: Usuario autenticado que inicia la retransmisión.

    Returns:
        RetransmisionResponse con los datos de la retransmisión creada.
    """
    token = generate_token()
    # Bucle de seguridad anti-colisión: regenera si el token ya existe en BD
    while db.query(Retransmision).filter(Retransmision.token == token).first():
        token = generate_token()

    nueva_retransmision = Retransmision(
        token=token,
        username=current_user.username,
        blancas=datos.blancas,
        negras=datos.negras,
        resultado=datos.resultado,
        evento=datos.evento,
        ronda=datos.ronda,
        tablero=datos.tablero,
        lugar=datos.lugar,
        is_activa=True
    )

    db.add(nueva_retransmision)
    db.commit()
    db.refresh(nueva_retransmision)

    return nueva_retransmision

@router.get("/status/{token}", summary="Obtener el estado de una retransmisión por token")
async def get_retransmision_status(token: str):
    """Obtiene el estado actual de una retransmisión por su token.

    Endpoint público que permite consultar si una retransmisión
    está activa y cuántos espectadores tiene conectados, útil
    para que el frontend decida si conectarse al WebSocket.

    Args:
        token: Token identificador de la retransmisión.

    Returns:
        Dict con el estado (activa/inactiva) y número de espectadores.
    """
    is_active = token in manager.hosts or token in manager.states
    viewers_count = len(manager.viewers.get(token, []))
    
    return {
        "success": True,
        "data": {
            "token": token,
            "active": is_active,
            "viewers": viewers_count
        }
    }

@router.patch(
    "/{id_retransmision}",
    summary="Actualizar retransmisión (ej. finalizar)",
)
def update_retransmision(
    id_retransmision: int,
    datos: RetransmisionCreate,
    db: Session = Depends(get_db),
    current_user: Usuario = Depends(get_current_user)
):
    retransmision = db.query(Retransmision).filter(
        Retransmision.id_retransmision == id_retransmision,
        Retransmision.username == current_user.username
    ).first()

    if not retransmision:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retransmisión no encontrada o no tienes permiso."
        )

    update_data = datos.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(retransmision, key, value)

    db.commit()
    db.refresh(retransmision)
    return retransmision

@router.websocket("/ws/host/{token}")
async def websocket_host(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """WebSocket para el emisor (host) de la retransmisión.

    Recibe el estado del tablero en formato JSON desde el emisor
    (generado por el módulo de visión) y lo reenvía a todos los
    espectadores conectados a la sala. Al desconectarse, finaliza
    la retransmisión y limpia los recursos.

    Args:
        websocket: Conexión WebSocket del host.
        token: Token identificador de la retransmisión.
        db: Sesión de base de datos.
    """
    await manager.connect_host(websocket, token)
    try:
        while True:
            # El host envía el estado actualizado del tablero (JSON)
            data = await websocket.receive_json()
            await manager.broadcast_to_viewers(token, data)
    except WebSocketDisconnect:
        await manager.disconnect_host(token, db)

@router.websocket("/ws/viewer/{token}")
async def websocket_viewer(websocket: WebSocket, token: str):
    """WebSocket público para los espectadores de una retransmisión.

    Conecta al espectador a la sala identificada por el token para
    recibir actualizaciones en tiempo real del estado del tablero.
    Soporta heartbeat mediante mensajes 'ping'/'pong' para mantener
    la conexión activa.

    Args:
        websocket: Conexión WebSocket del espectador.
        token: Token identificador de la retransmisión.
    """
    await manager.connect_viewer(websocket, token)
    try:
        while True:
            # Los espectadores pueden enviar "ping" como heartbeat
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect_viewer(websocket, token)

# routers/retransmision.py
# Gestión de retransmisiones en tiempo real
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from typing import Dict, List
from pydantic import BaseModel
from sqlalchemy.orm import Session
import string
import random

from db.database import get_db
from models.usuarios import Usuario
from routers.auth import get_current_user
from schemas.retransmisiones import RetransmisionCreate, RetransmisionResponse
from models.retransmisiones import Retransmision

# Creación del Router de Retransmisiones
router = APIRouter(
    prefix="/retransmision",
    tags=["Retransmisión en tiempo real"],
    responses={404: {"description": "No encontrado"}},
)

# Generación de Token para la retransmisión
def generate_token(length=8):
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

# Estado de la Retransmisión
class RetransmisionStatus(BaseModel):
    token: str
    active: bool
    viewers: int

# Gestión de Conexiones
class ConnectionManager:
    def __init__(self):
        # lista de websockets de espectadores
        self.viewers: Dict[str, List[WebSocket]] = {}
        # último estado enviado (para enviar a nuevos espectadores inmediatamente)
        self.states: Dict[str, dict] = {}
        # websocket del emisor
        self.hosts: Dict[str, WebSocket] = {}

    # Conexión de Espectadores
    async def connect_viewer(self, websocket: WebSocket, token: str):
        await websocket.accept()
        if token not in self.viewers:
            self.viewers[token] = []
        self.viewers[token].append(websocket)
        # Envía el estado más reciente al espectador recién conectado
        if token in self.states:
            await websocket.send_json(self.states[token])

    # Desconexión de Espectadores
    def disconnect_viewer(self, websocket: WebSocket, token: str):
        if token in self.viewers and websocket in self.viewers[token]:
            self.viewers[token].remove(websocket)

    # Conexión de Emisores
    async def connect_host(self, websocket: WebSocket, token: str):
        await websocket.accept()
        self.hosts[token] = websocket
        if token not in self.viewers:
            self.viewers[token] = []

    # Desconexión de Emisores
    async def disconnect_host(self, token: str, db):
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

    # Envío de Mensajes a Espectadores
    async def broadcast_to_viewers(self, token: str, message: dict):
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

# Instancia del Gestor de Conexiones
manager = ConnectionManager()

# Endpoint para crear retransmisión
# POST /retransmision/host
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
    """
    Crea un registro de retransmisión en la base de datos y genera un token único.
    """
    token = generate_token()
    # Asegurarnos de que no hay colisión de token
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
        is_activa=True  # Se marca como activa desde el inicio
    )

    db.add(nueva_retransmision)
    db.commit()
    db.refresh(nueva_retransmision)

    return nueva_retransmision

# Endpoint para obtener estado de la retransmisión
# GET /retransmision/status/{token}
@router.get("/status/{token}", summary="Obtener el estado de una retransmisión por token")
async def get_retransmision_status(token: str):
    """
    Endpoint público para consultar si una retransmisión está activa
    antes de conectarse al WebSocket.
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

# WebSocket para el emisor de la retransmisión
# WS /retransmision/ws/host/{token}
@router.websocket("/ws/host/{token}")
async def websocket_host(websocket: WebSocket, token: str, db: Session = Depends(get_db)):
    """
    WebSocket para el emisor de la retransmisión.
    Recibe el estado del tablero (generado por vision.py) y lo reenvía a los espectadores.
    """
    await manager.connect_host(websocket, token)
    try:
        while True:
            # El host envía el estado actualizado del tablero (JSON)
            data = await websocket.receive_json()
            await manager.broadcast_to_viewers(token, data)
    except WebSocketDisconnect:
        await manager.disconnect_host(token, db)

# WebSocket para los espectadores de la retransmisión
# WS /retransmision/ws/viewer/{token}
@router.websocket("/ws/viewer/{token}")
async def websocket_viewer(websocket: WebSocket, token: str):
    """
    WebSocket público para los espectadores de una retransmisión.
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

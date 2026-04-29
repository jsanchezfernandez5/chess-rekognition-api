from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
from pydantic import BaseModel

router = APIRouter(
    prefix="/retransmision",
    tags=["Retransmisión"],
    responses={404: {"description": "No encontrado"}},
)

class RetransmisionStatus(BaseModel):
    token: str
    active: bool
    viewers: int

# Memory storage for active broadcasts
class ConnectionManager:
    def __init__(self):
        # token -> list of viewer websockets
        self.viewers: Dict[str, List[WebSocket]] = {}
        # token -> last broadcasted state (to send to new viewers immediately)
        self.states: Dict[str, dict] = {}
        # token -> host websocket (optional, mainly to track if host is alive)
        self.hosts: Dict[str, WebSocket] = {}

    async def connect_viewer(self, websocket: WebSocket, token: str):
        await websocket.accept()
        if token not in self.viewers:
            self.viewers[token] = []
        self.viewers[token].append(websocket)
        # Send the latest state to the newly connected viewer
        if token in self.states:
            await websocket.send_json(self.states[token])

    def disconnect_viewer(self, websocket: WebSocket, token: str):
        if token in self.viewers and websocket in self.viewers[token]:
            self.viewers[token].remove(websocket)

    async def connect_host(self, websocket: WebSocket, token: str):
        await websocket.accept()
        self.hosts[token] = websocket
        if token not in self.viewers:
            self.viewers[token] = []

    def disconnect_host(self, token: str):
        if token in self.hosts:
            del self.hosts[token]

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

manager = ConnectionManager()

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

@router.websocket("/ws/host/{token}")
async def websocket_host(websocket: WebSocket, token: str):
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
        manager.disconnect_host(token)


@router.websocket("/ws/viewer/{token}")
async def websocket_viewer(websocket: WebSocket, token: str):
    """
    WebSocket público para los espectadores de una retransmisión.
    """
    await manager.connect_viewer(websocket, token)
    try:
        while True:
            # Los espectadores normalmente no envían datos, pero mantenemos la conexión abierta
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_viewer(websocket, token)

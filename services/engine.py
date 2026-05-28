# services/engine.py
# Lógica de negocio relacionada con el motor de ajedrez Stockfish.
"""Módulo de integración con el motor de ajedrez Stockfish.

Proporciona la clase StockfishService que gestiona la comunicación con el binario
de Stockfish mediante subprocesos y el protocolo UCI. Ofrece métodos para verificar
el estado del motor y calcular la mejor jugada dada una posición FEN."""
import os
import subprocess
import pathlib
import platform
from typing import Optional, Tuple, TypedDict

# Clases internas para el tipado.

# Tipos para la respuesta del motor.
class ScoreType(TypedDict):
    """Tipo estructurado que representa la puntuación calculada por el motor.

    Attributes:
        type: Tipo de puntuación ("cp" para centipeones, "mate" para mate).
        value: Valor numérico de la puntuación.
    """
    type: str
    value: int

# Info del análisis del motor.
class EngineInfo(TypedDict):
    """Tipo estructurado con la información detallada del análisis del motor.

    Attributes:
        score: Puntuación de la posición (puede ser None si no se ha calculado).
        depth: Profundidad de búsqueda en semimovidas alcanzada por el motor.
        nodes: Número de nodos explorados durante el análisis.
        pv: Línea principal (principal variation) como cadena de movimientos UCI.
    """
    score: Optional[ScoreType]
    depth: int
    nodes: int
    pv: str

# Status de la respuesta del motor.
class StatusResponse(TypedDict, total=False):
    """Tipo estructurado para la respuesta de verificación de estado del motor.

    Attributes:
        status: Estado del motor ("ok" o "error").
        message: Mensaje descriptivo del resultado de la verificación.
        engine: Información de versión del motor Stockfish.
    """
    status: str
    message: str
    engine: str

# Clase principal para el motor de ajedrez.
# Gestiona la comunicación con el binario de Stockfish.
class StockfishService:
    def __init__(self):
        """Inicializa el servicio de Stockfish determinando la ruta del binario según el sistema operativo.

        Selecciona el binario adecuado entre las versiones Linux y Windows almacenadas
        en el directorio engine/. En sistemas Linux, aplica permisos de ejecución (755)
        al binario si es necesario.
        """
        # Determinamos la ruta del binario según el SO
        current_dir = pathlib.Path(__file__).parent.parent
        self.stockfish_path = os.path.join(current_dir, "engine", "stockfish-linux-17.1")
        
        if platform.system() == "Windows":
            self.stockfish_path = os.path.join(current_dir, "engine", "stockfish-windows-17.1.exe")
        
        # Aseguramos permisos de ejecución en Linux
        if platform.system() != "Windows" and os.path.exists(self.stockfish_path):
            try:
                # 755 = rwxr-xr-x
                os.chmod(self.stockfish_path, int('755', 8))
            except Exception as e:
                print(f"Aviso: No se pudieron aplicar permisos al motor: {e}")

    def check_status(self) -> StatusResponse:
        """Verifica que el binario de Stockfish existe y responde correctamente al protocolo UCI.

        Lanza un subproceso con el binario, envía los comandos "uci" y "quit",
        y comprueba que la respuesta contiene la línea "uciok". Incluye un timeout
        de 5 segundos para evitar bloqueos indefinidos.

        Returns:
            StatusResponse con status "ok" e información de versión si el motor
            responde correctamente. En caso contrario, status "error" con un mensaje
            descriptivo del fallo.
        """
        if not os.path.exists(self.stockfish_path):
            return {"status": "error", "message": "Binario no encontrado"}
        
        try:
            # Usar communicate con timeout para evitar bloqueos indefinidos
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                outs, _ = process.communicate(input="uci\nquit\n", timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                return {"status": "error", "message": "Timeout al comunicar con el motor"}
            
            uciok = False
            version = "Stockfish"
            
            for line in outs.splitlines():
                if "Stockfish" in line:
                    version = line
                if line == "uciok": 
                    uciok = True
                    break
            
            if uciok:
                return {"status": "ok", "engine": version}
            else:
                return {"status": "error", "message": "Fallo de respuesta UCI"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_best_move(self, fen: str, elo: Optional[int] = None, depth: int = 15) -> Tuple[Optional[str], EngineInfo]:
        """Calcula la mejor jugada para una posición dada usando Stockfish.

        Comunica con el binario de Stockfish mediante el protocolo UCI, enviando
        la posición FEN y los parámetros de búsqueda. Si se proporciona un ELO,
        limita la fuerza del motor mediante UCI_LimitStrength para adaptarse al
        nivel del jugador. Parsea la salida para extraer la mejor jugada y los
        datos de análisis (puntuación, profundidad, nodos, línea principal).

        Args:
            fen: Notación FEN de la posición actual del tablero.
            elo: Puntuación ELO del jugador para limitar la fuerza del motor.
                 Si es None, el motor juega a máxima capacidad.
            depth: Profundidad de búsqueda en semimovidas (por defecto 15).

        Returns:
            Tupla (best_move, info) donde best_move es la jugada UCI o None si no
            se encontró, e info es un diccionario EngineInfo con los detalles del análisis.

        Raises:
            FileNotFoundError: Si el binario de Stockfish no existe en la ruta esperada.
            TimeoutError: Si el motor excede el tiempo máximo de respuesta (20 segundos).
        """
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError(f"El binario de Stockfish no se encuentra en {self.stockfish_path}")

        process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            commands = "uci\nsetoption name Threads value 1\n"
            
            if elo is not None:
                # Mapeo aproximado de ELO a Skill Level de Stockfish (0-20)
                # Stockfish asume que Skill Level 0 es aprox ELO 1320 y Skill Level 20 es ELO 3190
                # Esta fórmula es la interpolación lineal estándar usada por la comunidad UCI.
                MIN_ELO = 1320
                MAX_ELO = 3190
                SKILL_LEVELS = 20
                
                skill = int(round((elo - MIN_ELO) / ((MAX_ELO - MIN_ELO) / SKILL_LEVELS)))
                skill = max(0, min(skill, 20))
                
                commands += "setoption name UCI_LimitStrength value true\n"
                commands += f"setoption name UCI_Elo value {elo}\n"
                commands += f"setoption name Skill Level value {skill}\n"
            
            commands += "ucinewgame\n"
            commands += f"position fen {fen}\n"
            commands += f"go depth {depth}\n"
            
            try:
                # Tiempo de espera máximo calculado aprox según profundidad (15 segundos max de seguridad)
                outs, _ = process.communicate(input=commands, timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError("El motor excedió el tiempo máximo de respuesta.")

            best_move = None
            info: EngineInfo = {
                "score": None,
                "depth": 0,
                "nodes": 0,
                "pv": ""
            }

            # Parseo de salida
            for line in outs.splitlines():
                line = line.strip()
                if not line: continue
                
                if line.startswith("info "):
                    parts = line.split(" ")
                    if "depth" in parts:
                        idx = parts.index("depth")
                        info["depth"] = int(parts[idx+1])
                    if "nodes" in parts:
                        idx = parts.index("nodes")
                        info["nodes"] = int(parts[idx+1])
                    if "score" in parts:
                        idx = parts.index("score")
                        score_type = parts[idx+1] # cp o mate
                        score_val = parts[idx+2]
                        info["score"] = {"type": score_type, "value": int(score_val)}
                    if "pv" in parts:
                        idx = parts.index("pv")
                        info["pv"] = " ".join(parts[idx+1:])

                if line.startswith("bestmove"):
                    parts = line.split(" ")
                    best_move = parts[1]

            return best_move, info

        finally:
            # Asegurarse siempre de matar el proceso por seguridad,
            # communicate() cierra stdin/out y espera al exit code, pero por si acaso.
            if process.poll() is None:
                process.kill()

# Instancia global del servicio.
# Uso: from services.engine import engine_service
engine_service = StockfishService()

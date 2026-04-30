# services/engine.py
# Lógica de negocio relacionada con el motor de ajedrez Stockfish.
import os
import subprocess
import pathlib
import platform
from typing import Optional, Tuple, TypedDict

# Clases internas para el tipado.

# Tipos para la respuesta del motor.
class ScoreType(TypedDict):
    type: str
    value: int

# Info del análisis del motor.
class EngineInfo(TypedDict):
    score: Optional[ScoreType]
    depth: int
    nodes: int
    pv: str

# Status de la respuesta del motor.
class StatusResponse(TypedDict, total=False):
    status: str
    message: str
    engine: str

# Clase principal para el motor de ajedrez.
# Gestiona la comunicación con el binario de Stockfish.
class StockfishService:
    # Constructor. 
    def __init__(self):
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

    # Método para verificar el estado del motor.
    def check_status(self) -> StatusResponse:
        """
        Verifica de forma rápida que el binario existe y responde a comandos UCI.
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

    # Método principal para obtener la mejor jugada.
    # Calcula la mejor jugada según la posición y el ELO del jugador.
    def get_best_move(self, fen: str, elo: Optional[int] = None, depth: int = 15) -> Tuple[Optional[str], EngineInfo]:
        """
        Llama al binario de Stockfish para obtener la mejor jugada y datos de análisis.
        Retorna (best_move, info_dict)
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

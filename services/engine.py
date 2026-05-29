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


class ScoreType(TypedDict):
    """Tipo estructurado que representa la puntuación calculada por el motor.

    Attributes:
        type: Tipo de puntuación ("cp" para centipeones, "mate" para mate).
        value: Valor numérico de la puntuación.
    """
    type: str
    value: int

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

class StockfishService:
    def __init__(self):
        """Inicializa el servicio de Stockfish con la ruta al binario según el SO."""
        current_dir = pathlib.Path(__file__).parent.parent
        self.stockfish_path = os.path.join(current_dir, "engine", "stockfish-linux-17.1")
        
        if platform.system() == "Windows":
            self.stockfish_path = os.path.join(current_dir, "engine", "stockfish-windows-17.1.exe")
        
        # Permisos de ejecución necesarios en Linux
        if platform.system() != "Windows" and os.path.exists(self.stockfish_path):
            try:
                os.chmod(self.stockfish_path, int('755', 8))
            except Exception as e:
                print(f"Aviso: No se pudieron aplicar permisos al motor: {e}")

    def check_status(self) -> StatusResponse:
        """Verifica que Stockfish responde al protocolo UCI correctamente.

        Lanza un subproceso, envía "uci"/"quit" y busca "uciok" en la respuesta.

        Returns:
            StatusResponse con "ok" y versión, o "error" con mensaje.
        """
        if not os.path.exists(self.stockfish_path):
            return {"status": "error", "message": "Binario no encontrado"}
        
        try:
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True
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
            return {"status": "error", "message": "Fallo de respuesta UCI"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_best_move(
        self, fen: str, elo: Optional[int] = None, depth: int = 15
    ) -> Tuple[Optional[str], EngineInfo]:
        """Calcula la mejor jugada para una posición FEN usando Stockfish.

        Se comunica con el binario mediante protocolo UCI. Si se proporciona ELO,
        limita la fuerza con UCI_LimitStrength. Parsea la salida para extraer
        la jugada y métricas de análisis.

        Args:
            fen: Posición en notación FEN.
            elo: ELO para limitar fuerza (None = máxima capacidad).
            depth: Profundidad de búsqueda en semimovidas.

        Returns:
            Tupla (best_move, info) con jugada UCI y detalles del análisis.

        Raises:
            FileNotFoundError: Si el binario no existe.
            TimeoutError: Si el motor excede 20 segundos.
        """
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError(f"Binario no encontrado en {self.stockfish_path}")

        process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True
        )

        try:
            commands = "uci\nsetoption name Threads value 1\n"

            if elo is not None:
                # Interpolación lineal ELO → Skill Level (0-20)
                # Rango Stockfish: 1320 (skill 0) a 3190 (skill 20)
                MIN_ELO, MAX_ELO, SKILL_LEVELS = 1320, 3190, 20
                skill = int(round((elo - MIN_ELO) / ((MAX_ELO - MIN_ELO) / SKILL_LEVELS)))
                skill = max(0, min(skill, 20))

                commands += "setoption name UCI_LimitStrength value true\n"
                commands += f"setoption name UCI_Elo value {elo}\n"
                commands += f"setoption name Skill Level value {skill}\n"

            commands += "ucinewgame\n"
            commands += f"position fen {fen}\n"
            commands += f"go depth {depth}\n"

            try:
                outs, _ = process.communicate(input=commands, timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError("El motor excedió el tiempo máximo de respuesta.")

            best_move = None
            info: EngineInfo = {"score": None, "depth": 0, "nodes": 0, "pv": ""}

            # Parseo de la salida UCI: busca líneas "info" con métricas y "bestmove" con jugada
            for line in outs.splitlines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("info "):
                    parts = line.split(" ")
                    # Extrae profundidad, nodos y puntuación por palabras clave
                    if "depth" in parts:
                        idx = parts.index("depth")
                        info["depth"] = int(parts[idx + 1])
                    if "nodes" in parts:
                        idx = parts.index("nodes")
                        info["nodes"] = int(parts[idx + 1])
                    if "score" in parts:
                        idx = parts.index("score")
                        score_type = parts[idx + 1]  # "cp" (centipeones) o "mate"
                        score_val = parts[idx + 2]
                        info["score"] = {"type": score_type, "value": int(score_val)}
                    if "pv" in parts:
                        idx = parts.index("pv")
                        info["pv"] = " ".join(parts[idx + 1:])

                if line.startswith("bestmove"):
                    parts = line.split(" ")
                    best_move = parts[1]

            return best_move, info

        finally:
            if process.poll() is None:
                process.kill()


engine_service = StockfishService()

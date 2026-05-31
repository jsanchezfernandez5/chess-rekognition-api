"""
Servicio de integración con el motor de ajedrez Stockfish.
"""
import os
import platform
import pathlib
import threading
import logging

import chess
from stockfish import Stockfish

# Log
logger = logging.getLogger("services.engine")

# Ruta al binario
_BASE_DIR = pathlib.Path(__file__).parent.parent

# En producción (Railway/Linux) se usa el binario Linux; en local Windows el .exe
_STOCKFISH_PATH = str(
    _BASE_DIR / "engine" / (
        "stockfish-windows-17.1.exe"
        if platform.system() == "Windows"
        else "stockfish-linux-17.1"
    )
)

# Parámetros base del motor
_ENGINE_PARAMS = {
    "Threads": 1,
    "Move Overhead": 30,
}

# Lock global: evita llamadas concurrentes al binario
_ENGINE_LOCK = threading.Lock()

# Tabla ELO → Skill Level. Mapeo no lineal calibrado para Stockfish. Skill Level 0-20.
# UCI_LimitStrength NO se usa porque `go depth` lo anula.
_ELO_SKILL_TABLE = [
    (1320, 0),  (1400, 1),  (1500, 2),  (1600, 4),
    (1700, 5),  (1800, 7),  (1900, 9),  (2000, 10),
    (2100, 12), (2200, 13), (2300, 14), (2400, 15),
    (2500, 16), (2600, 17), (2700, 18), (2800, 19),
    (2900, 19), (3000, 20), (3190, 20),
]

# Función para convertir ELO a Skill Level
def _elo_to_skill(elo: int) -> int:
    """
    Convierte un valor ELO (1320-3190) al Skill Level correspondiente (0-20).
    Usa interpolación lineal entre los puntos de la tabla calibrada.
    """
    if elo <= _ELO_SKILL_TABLE[0][0]:
        return _ELO_SKILL_TABLE[0][1]
    if elo >= _ELO_SKILL_TABLE[-1][0]:
        return _ELO_SKILL_TABLE[-1][1]

    for i in range(1, len(_ELO_SKILL_TABLE)):
        if elo < _ELO_SKILL_TABLE[i][0]:
            elo_low,  skill_low  = _ELO_SKILL_TABLE[i - 1]
            elo_high, skill_high = _ELO_SKILL_TABLE[i]
            ratio = (elo - elo_low) / (elo_high - elo_low)
            return round(skill_low + ratio * (skill_high - skill_low))

    return _ELO_SKILL_TABLE[-1][1]

# Función para crear una instancia de Stockfish
def _create_engine() -> Stockfish | None:
    """
    Crea y devuelve una instancia fresca de Stockfish, o None si falla.
    """
    if not os.path.isfile(_STOCKFISH_PATH):
        logger.error(f"Binario no encontrado: {_STOCKFISH_PATH}")
        return None

    # En Linux aplicamos permisos de ejecución por si el binario llegó sin ellos
    if platform.system() != "Windows":
        try:
            os.chmod(_STOCKFISH_PATH, 0o755)
        except Exception as e:
            logger.warning(f"No se pudieron aplicar permisos: {e}")

    try:
        # Retorna una instancia de Stockfish
        eng = Stockfish(path=_STOCKFISH_PATH, parameters=_ENGINE_PARAMS)
        return eng
    except Exception as e:
        logger.error(f"No se pudo inicializar Stockfish: {e}")
        return None

# Función para cerrar el proceso de Stockfish de forma segura
def _safe_cleanup(eng: Stockfish | None) -> None:
    """
    Cierra el proceso de Stockfish de forma segura.
    """
    try:
        if eng and hasattr(eng, "_stockfish") and eng._stockfish is not None:
            process = eng._stockfish
            if process.poll() is None:
                try:
                    eng.send_quit_command()
                    process.wait(timeout=1.0)
                except Exception:
                    pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.5)
                    except Exception:
                        process.kill()
                        process.wait()
    except Exception as e:
        logger.error(f"Error limpiando proceso Stockfish: {e}")

# Clase principal
class StockfishService:

    # Método para verificar el estado del binario
    def check_status(self) -> dict:
        """
        Verifica que el binario de Stockfish existe y es accesible.

        Returns:
            Dict con status 'ok' o 'error' y detalles.
        """
        if not os.path.isfile(_STOCKFISH_PATH):
            return {
                "status": "error",
                "engine": "Stockfish",
                "binary_exists": False,
                "detail": f"Binario no encontrado en {_STOCKFISH_PATH}",
            }
        return {
            "status": "ok",
            "engine": "Stockfish 17.1",
            "binary_exists": True,
            "path": _STOCKFISH_PATH,
        }

    # Método para obtener la mejor jugada
    def get_best_move(
        self,
        fen: str,
        elo: int | None = None,
        depth: int = 15,
    ) -> tuple[str | None, dict]:
        """
        Calcula la mejor jugada para una posición FEN.

        Args:
            fen:   Posición en notación FEN.
            elo:   ELO deseado (1320-3190). None = fuerza máxima.
            depth: Profundidad de búsqueda (1-30).

        Returns:
            Tupla (best_move, info). 
            Si la partida ya terminó, best_move es None e info contiene 'game_over', 'result' y 'reason'.

        Raises:
            RuntimeError: Si el motor no está disponible.
        """
        # Validación básica del FEN
        if not fen or any(c in fen for c in ("\n", "\r", ";")):
            raise ValueError("FEN no válido")

        # Detección de partida terminada antes de llamar al motor
        board = chess.Board(fen)
        if board.is_game_over():
            if board.is_checkmate():
                reason = "checkmate"
            elif board.is_stalemate():
                reason = "stalemate"
            elif board.is_insufficient_material():
                reason = "insufficient_material"
            else:
                reason = "draw"
            return None, {"game_over": True, "result": board.result(), "reason": reason}

        with _ENGINE_LOCK:
            eng = _create_engine()
            if eng is None:
                raise RuntimeError("Motor Stockfish no disponible")

            try:
                eng.set_fen_position(fen)

                # Nivel de dificultad: solo Skill Level, nunca UCI_LimitStrength
                if elo is not None:
                    elo = max(1320, min(int(elo), 3190))
                    eng.set_skill_level(_elo_to_skill(elo))
                else:
                    # Fuerza máxima: Skill Level 20
                    eng.set_skill_level(20)

                # Profundidad de búsqueda
                depth = max(1, min(int(depth), 30))
                eng.set_depth(depth)

                # Recogemos los parámetros: best_move, evaluation y top_moves
                best_move  = eng.get_best_move()
                evaluation = eng.get_evaluation()
                top_moves  = eng.get_top_moves(3)

                # Retornamos la mejor jugada y la información adicional
                return best_move, {
                    "evaluation": evaluation,
                    "top_moves":  top_moves,
                    "depth":      depth,
                    "skill_level": _elo_to_skill(elo) if elo else 20,
                }

            finally:
                # Cerrar el proceso de Stockfish de forma segura
                _safe_cleanup(eng)

# Instancia global utilizada por el router
engine_service = StockfishService()

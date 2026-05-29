"""Módulo de detección de movimientos de ajedrez mediante comparación visual.

Compara el estado del tablero clasificado por el modelo ML con la posición anterior
representada en FEN para identificar el movimiento legal realizado. Incluye funciones
auxiliares para clasificar el tipo de movimiento y calcular la confianza media."""
import chess
import numpy as np
from utils.chess_utils import label_to_piece

def detect_move(warped: np.ndarray, prev_fen: str, classifier) -> dict:
    """Detecta el movimiento realizado comparando el estado visual actual con el FEN anterior.

    Clasifica el tablero actual mediante el clasificador ML, construye el estado
    de piezas, y busca entre todos los movimientos legales de la posición anterior
    aquel cuya posición resultante coincide con el estado clasificado.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.
        prev_fen: Notación FEN de la posición anterior al movimiento.
        classifier: Instancia de ChessClassifier para clasificar el tablero.

    Returns:
        Diccionario con los resultados. Si se encuentra el movimiento, incluye
        found=True, move (con uci, san, from, to, promotion, type), new_fen,
        board_state y confidence_avg. Si no se encuentra, found=False con
        error descriptivo.
    """
    try:
        board_state = classifier.classify_board(warped)
        simple_state = {sq: v["label"] for sq, v in board_state.items()}

        try:
            prev_board = chess.Board(prev_fen)
        except ValueError:
            return {"found": False, "error": "fen_invalid"}

        # Busca entre todos los movimientos legales de la posición anterior
        # aquel que genera una disposición de piezas igual a la clasificada por ML
        for move in prev_board.legal_moves:
            test = prev_board.copy()
            test.push(move)

            if _positions_match(test, simple_state):
                move_type = _classify_move_type(prev_board, move)
                return {
                    "found": True,
                    "move": {
                        "uci": move.uci(),
                        "san": prev_board.san(move),
                        "from": chess.square_name(move.from_square),
                        "to": chess.square_name(move.to_square),
                        "promotion": chess.piece_name(move.promotion) if move.promotion else None,
                        "type": move_type
                    },
                    "new_fen": test.fen(),
                    "board_state": board_state,
                    "confidence_avg": _avg_confidence(board_state)
                }

        return {
            "found": False,
            "error": "no_legal_move_found",
            "board_state": board_state,
            "confidence_avg": _avg_confidence(board_state)
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

def _positions_match(board: chess.Board, simple_state: dict) -> bool:
    """Compara cada casilla del Board con la clasificación ML."""
    for square in chess.SQUARES:
        sq_name = chess.square_name(square)
        piece_on_board = board.piece_at(square)
        label_from_ml = simple_state.get(sq_name)
        
        piece_from_ml = label_to_piece(label_from_ml) if label_from_ml else None
        
        if piece_on_board != piece_from_ml:
            return False
            
    return True

def _classify_move_type(board: chess.Board, move: chess.Move) -> str:
    """Clasifica el tipo de movimiento: enroque, captura, promoción, etc."""
    if board.is_castling(move):
        # Diferenciar entre corto y largo
        if board.is_kingside_castling(move):
            return "castling_short"
        else:
            return "castling_long"
            
    if board.is_en_passant(move):
        return "en_passant"
        
    if move.promotion:
        return "promotion"
        
    if board.is_capture(move):
        return "capture"
        
    return "normal"

def _avg_confidence(board_state: dict) -> float:
    """Calcula la confianza media del clasificador ML en todas las casillas."""
    confidences = [v["confidence"] for v in board_state.values()]
    if not confidences:
        return 0.0
    return round(float(np.mean(confidences)), 3)

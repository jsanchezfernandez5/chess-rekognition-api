import chess
import numpy as np
from utils.chess_utils import label_to_piece

def detect_move(warped: np.ndarray, prev_fen: str, classifier) -> dict:
    """
    Compara el estado visual actual con el FEN anterior y detecta qué movimiento legal se ha realizado.
    """
    try:
        # 1. Obtener el estado ML del tablero
        board_state = classifier.classify_board(warped)
        
        # 2. Construir el simple_state (solo etiquetas)
        simple_state = {sq: v["label"] for sq, v in board_state.items()}
        
        # 3. Cargar el estado anterior con python-chess
        try:
            prev_board = chess.Board(prev_fen)
        except ValueError:
            return {"found": False, "error": "fen_invalid"}

        # 4. Buscar el movimiento legal que coincide con la nueva posición
        for move in prev_board.legal_moves:
            test = prev_board.copy()
            test.push(move)
            
            if _positions_match(test, simple_state):
                # Encontrado
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

        # No se encontró ningún movimiento legal que resulte en esta posición
        return {
            "found": False,
            "error": "no_legal_move_found",
            "board_state": board_state,
            "confidence_avg": _avg_confidence(board_state)
        }
    except Exception as e:
        return {"found": False, "error": str(e)}

def _positions_match(board: chess.Board, simple_state: dict) -> bool:
    """
    Compara el placement de piezas de un chess.Board con el output ML.
    """
    for square in chess.SQUARES:
        sq_name = chess.square_name(square)
        piece_on_board = board.piece_at(square)
        label_from_ml = simple_state.get(sq_name)
        
        piece_from_ml = label_to_piece(label_from_ml) if label_from_ml else None
        
        if piece_on_board != piece_from_ml:
            return False
            
    return True

def _classify_move_type(board: chess.Board, move: chess.Move) -> str:
    """
    Determina el tipo de movimiento realizado.
    """
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
    """Calcula la confianza media de las 64 casillas."""
    confidences = [v["confidence"] for v in board_state.values()]
    if not confidences:
        return 0.0
    return round(float(np.mean(confidences)), 3)

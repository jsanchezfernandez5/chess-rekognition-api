"""
Módulo de detección de movimientos de ajedrez mediante comparación visual.

Compara el estado del tablero clasificado por el modelo ML con la posición anterior representada en FEN.
"""
import chess
import numpy as np

from services.vision import label_to_piece

# Función para detectar movimientos
def detect_move(warped: np.ndarray, prev_fen: str, classifier) -> dict:
    """
    Detecta el movimiento realizado comparando el estado visual actual con el FEN anterior.

    Args:
        warped: Imagen del tablero rectificado en vista cenital.
        prev_fen: Notación FEN de la posición anterior al movimiento.
        classifier: Instancia de ChessClassifier para clasificar el tablero.

    Returns:
        Diccionario con los resultados.
    """
    try:
        # Clasificamos el tablero actual mediante el clasificador ML
        board_state = classifier.classify_board(warped)

        # Convertimos el estado del tablero a un formato simple
        simple_state = {sq: v["label"] for sq, v in board_state.items()}

        # Obtenemos la posición anterior desde el FEN proporcionado
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

# Función para comparar cada casilla del Board con la clasificación ML
def _positions_match(board: chess.Board, simple_state: dict) -> bool:
    """
    Compara cada casilla del Board con la clasificación ML.
    
    Args:
        board: Objeto chess.Board con la posición actual.
        simple_state: Diccionario con la clasificación ML de cada casilla.

    Returns:
        True si las posiciones coinciden, False en caso contrario.
    """
    for square in chess.SQUARES:
        # Obtenemos el nombre de la casilla (ej. "a1")
        sq_name = chess.square_name(square)
        
        # Obtenemos la pieza en la casilla del tablero
        piece_on_board = board.piece_at(square)
        
        # Obtenemos la etiqueta de la pieza desde la clasificación ML
        label_from_ml = simple_state.get(sq_name)
        
        # Obtenemos la pieza desde la clasificación ML
        piece_from_ml = label_to_piece(label_from_ml) if label_from_ml else None
        
        if piece_on_board != piece_from_ml:
            return False
            
    return True

def _classify_move_type(board: chess.Board, move: chess.Move) -> str:
    """
    Clasifica el tipo de movimiento: enroque, captura, promoción, etc.
    
    Args:
        board: Objeto chess.Board con la posición actual.
        move: Movimiento a clasificar.

    Returns:
        Tipo de movimiento: "castling_short", "castling_long", "en_passant", "promotion", "capture", "normal".
    """
    # Diferenciar entre corto y largo
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            return "castling_short"
        else:
            return "castling_long"
            
    # Captura al paso.
    if board.is_en_passant(move):
        return "en_passant"

    # Coronación  
    if move.promotion:
        return "promotion"

    # Captura
    if board.is_capture(move):
        return "capture"

    # Movimiento normal    
    return "normal"

# Función para calcular la confianza media
def _avg_confidence(board_state: dict) -> float:
    """
    Calcula la confianza media del clasificador ML en todas las casillas.
    
    Args:
        board_state: Diccionario con la clasificación ML de cada casilla.

    Returns:
        Confianza media del clasificador ML en todas las casillas.
    """
    confidences = [v["confidence"] for v in board_state.values()]
    if not confidences:
        return 0.0

    # Redondeamos a 3 decimales
    return round(float(np.mean(confidences)), 3)

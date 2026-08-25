"""
Servicio de detección de movimientos de ajedrez mediante comparación visual.

Compara el estado del tablero clasificado por el modelo YOLO con la posición anterior representada en FEN.

Funciones principales:
    - detect_move_desde_estado()     | Busca el movimiento legal que explica la transición desde prev_fen hacia un board_state ya clasificado por YOLO.
    - _positions_match()             | Función interna para comparar cada casilla del chess.Board con la clasificación ML.
    - _classify_move_type()          | Función interna que clasifica el tipo de movimiento.
    - _avg_confidence()              | Función interna que calcula la confianza media del detector YOLO en las 64 casillas.
"""
import chess
import numpy as np

from services.vision import label_to_piece

def detect_move_desde_estado(board_state: dict, prev_fen: str) -> dict:
    """
    Busca el movimiento legal que explica la transición desde prev_fen hacia un board_state ya clasificado.

    El resultado depende solo del estado final clasificado, no de qué motor lo produjo.

    Args:
        board_state: Diccionario {"e4": {"label": "w_P", "confidence": 0.91}, ...} ya clasificado.
        prev_fen: Notación FEN de la posición anterior al movimiento.

    Returns:
        Diccionario con found/move/new_fen/error/board_state/confidence_avg.
    """
    try:
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

def _avg_confidence(board_state: dict) -> float:
    """
    Calcula la confianza media del detector YOLO en todas las casillas.
    
    Args:
        board_state: Diccionario con la clasificación de cada casilla.

    Returns:
        Confianza media del detector en todas las casillas.
    """
    # Confianzas None (casillas vacías de YOLO, sin valor numérico) se excluyen de la media
    confidences = [v["confidence"] for v in board_state.values() if v.get("confidence") is not None]
    if not confidences:
        return 0.0

    # Redondeamos a 3 decimales
    return round(float(np.mean(confidences)), 3)

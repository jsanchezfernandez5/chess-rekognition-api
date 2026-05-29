"""
Utilidades de ajedrez para convertir entre etiquetas del clasificador,
estados de tablero y objetos de la librería chess.py.
"""
import chess

# Función para convertir una etiqueta del clasificador a un objeto chess.Piece
def label_to_piece(label: str) -> chess.Piece | None:
    """
    Convierte una etiqueta del clasificador de piezas a un objeto chess.Piece.

    Las etiquetas siguen el formato 'color_tipo', donde color es 'w' o 'b',
    y tipo es una letra mayúscula (P, N, B, R, Q, K). La etiqueta 'empty'
    representa una casilla vacía.

    Args:
        label: Etiqueta del clasificador (ej: 'w_P', 'b_N', 'empty').

    Returns:
        chess.Piece si la casilla contiene una pieza, None si está vacía.
    """
    if label == "empty":
        return None

    color = chess.WHITE if label.startswith("w_") else chess.BLACK
    symbol = label.split("_")[1]

    # En FEN, las blancas son mayúsculas y las negras minúsculas
    fen_symbol = symbol if color == chess.WHITE else symbol.lower()
    return chess.Piece.from_symbol(fen_symbol)

# Función para convertir un estado de tablero representado como un diccionario a un objeto chess.Board
def board_state_to_fen_board(board_state: dict) -> chess.Board:
    """
    Convierte un diccionario con el estado del tablero a un objeto chess.Board.

    El diccionario debe contener pares sq_id: label, donde sq_id es la
    notación algebraica de la casilla (ej: 'e2', 'a1') y label es la
    etiqueta del clasificador (ej: 'w_P', 'empty').

    Args:
        board_state: Diccionario con el estado de cada casilla.

    Returns:
        chess.Board con las piezas colocadas según el estado proporcionado.
    """
    board = chess.Board(fen=None)
    for sq_name, label in board_state.items():
        piece = label_to_piece(label)
        if piece:
            board.set_piece_at(chess.parse_square(sq_name), piece)
    return board

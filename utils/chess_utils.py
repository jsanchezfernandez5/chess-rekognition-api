import chess

def label_to_piece(label: str) -> chess.Piece | None:
    """
    Convierte una etiqueta del clasificador a un objeto chess.Piece.
    Ejemplo: 'w_P' -> chess.Piece(PAWN, WHITE)
    """
    if label == "empty":
        return None
    
    # El color es el primer carácter después de w_ o b_
    color = chess.WHITE if label.startswith("w_") else chess.BLACK
    symbol = label.split("_")[1] # P, N, B, R, Q, K
    
    # En FEN, las blancas son mayúsculas y las negras minúsculas
    fen_symbol = symbol if color == chess.WHITE else symbol.lower()
    return chess.Piece.from_symbol(fen_symbol)

def board_state_to_fen_board(board_state: dict) -> chess.Board:
    """
    Convierte un diccionario de estado del tablero (sq_id: label) a un objeto chess.Board.
    Ejemplo: {"e2": "w_P", "e4": "empty"}
    """
    board = chess.Board(fen=None) # Tablero vacío
    for sq_name, label in board_state.items():
        piece = label_to_piece(label)
        if piece:
            board.set_piece_at(chess.parse_square(sq_name), piece)
    return board

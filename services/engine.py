# services/engine.py
# Servicio para interactuar con el motor Stockfish
import subprocess
import os
import pathlib
import platform
from typing import Optional

class StockfishService:
    def __init__(self):
        # Obtenemos la ruta raíz del proyecto API
        current_dir = pathlib.Path(__file__).parent.parent
        self.stockfish_path = os.path.join(current_dir, "engine", "stockfish-linux-17.1")
        
        # Si estamos en Windows, intentamos usar el .exe si existe
        if platform.system() == "Windows":
            exe_path = self.stockfish_path + ".exe"
            if os.path.exists(exe_path):
                self.stockfish_path = exe_path
        else:
            # Permisos 755
            try:
                if os.path.exists(self.stockfish_path):
                    os.chmod(self.stockfish_path, int('755', 8))
            except Exception as e:
                print(f"Aviso: No se pudieron aplicar permisos al motor: {e}")

    def check_status(self) -> dict:
        """
        Verifica que el binario existe, tiene permisos y responde a comandos básicos UCI.
        """
        if not os.path.exists(self.stockfish_path):
            return {"status": "error", "message": f"Binario no encontrado en {self.stockfish_path}"}
        
        try:
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            process.stdin.write("uci\n")
            process.stdin.flush()
            
            version_line = ""
            uciok = False
            
            # Leemos las primeras líneas para ver si responde uciok
            for _ in range(20):
                line = process.stdout.readline().strip()
                if not line: break
                if "Stockfish" in line: version_line = line
                if line == "uciok":
                    uciok = True
                    break
            
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.terminate()
            
            if uciok:
                return {
                    "status": "ok",
                    "engine": version_line or "Stockfish",
                    "path": self.stockfish_path,
                    "platform": platform.system()
                }
            else:
                return {"status": "error", "message": "El motor no respondió uciok"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_best_move(self, fen: str, elo: Optional[int] = None, depth: int = 15) -> str:
        """
        Llama al binario de Stockfish para obtener la mejor jugada para una posición FEN.
        """
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError(f"El binario de Stockfish no se encuentra en: {self.stockfish_path}")

        # Configuración del proceso Popen
        process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        try:
            def send_command(cmd):
                process.stdin.write(cmd + "\n")
                process.stdin.flush()

            # Inicialización UCI
            send_command("uci")
            send_command("setoption name Threads value 1")
            send_command("setoption name Move Overhead value 30")

            # Configuración de nivel (ELO)
            if elo is not None:
                # Normalización del ELO según la lógica del legacy (1320-3190 -> Skill 0-20)
                skill = int(round((elo - 1320) / ((3190 - 1320) / 20)))
                skill = max(0, min(skill, 20))
                
                send_command("setoption name UCI_LimitStrength value true")
                send_command("setoption name UCI_Elo value " + str(elo))
                send_command("setoption name Skill Level value " + str(skill))
            else:
                send_command("setoption name UCI_LimitStrength value false")
                send_command("setoption name Skill Level value 20")

            # Preparar posición y buscar
            send_command("ucinewgame")
            send_command(f"position fen {fen}")
            send_command(f"go depth {depth}")

            # Lectura de la salida hasta encontrar bestmove
            best_move = None
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("bestmove"):
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        best_move = parts[1]
                    break

            return best_move

        finally:
            if process.poll() is None:
                send_command("quit")
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()

engine_service = StockfishService()

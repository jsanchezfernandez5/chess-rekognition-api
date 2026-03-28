import os
import subprocess
import pathlib
import platform
from typing import Optional

class StockfishService:
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

    def check_status(self) -> dict:
        """
        Verifica de forma rápida que el binario existe y responde a comandos UCI.
        """
        if not os.path.exists(self.stockfish_path):
            return {"status": "error", "message": "Binario no encontrado"}
        
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
            
            uciok = False
            version = "Stockfish"
            for _ in range(20):
                line = process.stdout.readline().strip()
                if not line: break
                if "Stockfish" in line: version = line
                if line == "uciok": 
                    uciok = True
                    break
            
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.terminate()
            
            if uciok:
                return {"status": "ok", "engine": version}
            else:
                return {"status": "error", "message": "Fallo de respuesta UCI"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_best_move(self, fen: str, elo: Optional[int] = None, depth: int = 15) -> tuple:
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
            text=True,
            bufsize=1
        )

        try:
            def send_command(cmd):
                process.stdin.write(cmd + "\n")
                process.stdin.flush()

            send_command("uci")
            for _ in range(30):
                line = process.stdout.readline().strip()
                if not line: break
                if line == "uciok": break

            send_command("setoption name Threads value 1")
            if elo is not None:
                skill = int(round((elo - 1320) / ((3190 - 1320) / 20)))
                skill = max(0, min(skill, 20))
                send_command("setoption name UCI_LimitStrength value true")
                send_command("setoption name UCI_Elo value " + str(elo))
                send_command("setoption name Skill Level value " + str(skill))
            
            send_command("ucinewgame")
            send_command(f"position fen {fen}")
            send_command(f"go depth {depth}")

            best_move = None
            info = {
                "score": None,
                "depth": 0,
                "nodes": 0,
                "pv": ""
            }

            while True:
                line = process.stdout.readline()
                if not line: break
                line = line.strip()
                if not line: continue
                
                # Parseo de info
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
                    break

            return best_move, info

        finally:
            if process.poll() is None:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.terminate()

engine_service = StockfishService()

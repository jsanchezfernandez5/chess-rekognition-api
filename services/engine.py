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
        traces = []
        def log(msg):
            print(f"[Engine] {msg}")
            traces.append(msg)

        log(f"Verificando motor en: {self.stockfish_path}")
        if not os.path.exists(self.stockfish_path):
            log("Error: Binario no encontrado físicamente.")
            return {"status": "error", "message": "Binario no encontrado", "traces": traces}
        
        try:
            log("Intentando iniciar proceso...")
            process = subprocess.Popen(
                [self.stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            log(f"Proceso iniciado (PID: {process.pid}). Enviando 'uci'...")
            process.stdin.write("uci\n")
            process.stdin.flush()
            
            version_line = ""
            uciok = False
            
            # Leemos las primeras líneas para ver si responde uciok
            for i in range(20):
                line = process.stdout.readline().strip()
                if not line: 
                    # Si no hay salida en stdout, revisamos stderr
                    err = process.stderr.readline().strip()
                    if err: log(f"STDERR: {err}")
                    continue
                
                log(f"Recibido: {line}")
                if "Stockfish" in line: version_line = line
                if line == "uciok":
                    uciok = True
                    break
            
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.terminate()
            
            if uciok:
                log(f"Motor listo: {version_line}")
                return {
                    "status": "ok",
                    "engine": version_line or "Stockfish",
                    "path": self.stockfish_path,
                    "platform": platform.system(),
                    "traces": traces
                }
            else:
                stderr_remaining = process.stderr.read().strip()
                if stderr_remaining: log(f"STDERR FINAL: {stderr_remaining}")
                log("El motor no respondió uciok tras varios intentos.")
                return {"status": "error", "message": "Fallo UCI", "traces": traces}
                
        except Exception as e:
            log(f"Excepción crítica: {str(e)}")
            return {"status": "error", "message": str(e), "traces": traces}

    def get_best_move(self, fen: str, elo: Optional[int] = None, depth: int = 15) -> tuple:
        """
        Llama al binario de Stockfish para obtener la mejor jugada para una posición FEN.
        Retorna (best_move, traces)
        """
        traces = []
        def log(msg):
            print(f"[Engine] {msg}")
            traces.append(msg)

        log(f"Solicitando jugada (ELO: {elo}, Depth: {depth}) para FEN: {fen}")
        if not os.path.exists(self.stockfish_path):
            log(f"Error: Binario no encontrado en {self.stockfish_path}")
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
                log(f"Enviando: {cmd}")
                process.stdin.write(cmd + "\n")
                process.stdin.flush()

            # Inicialización UCI
            send_command("uci")
            send_command("setoption name Threads value 1")
            
            # ... lectura inicial ...
            for _ in range(10):
                line = process.stdout.readline().strip()
                if line: log(f"Motor: {line}")
                if line == "uciok": break

            # Configuración de nivel (ELO)
            if elo is not None:
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
                    err = process.stderr.read().strip()
                    if err: log(f"STDERR: {err}")
                    break
                
                line = line.strip()
                if line: log(f"Motor: {line}")
                if line.startswith("bestmove"):
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        best_move = parts[1]
                    break

            return best_move, traces

        finally:
            if process.poll() is None:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.terminate()

engine_service = StockfishService()

# ═══════════════════════════════════════════════════════════════
# Dockerfile — Chess Rekognition API (Railway)
#
# ¿Por qué un Dockerfile después de dos intentos vía nixpacks.toml?
# La traza del crash mostraba Python en /mise/installs/python/3.13.15
# y el venv en /app/.venv: eso es RAILPACK, el builder por defecto de
# Railway actualmente, que NO lee nixpacks.toml. Los intentos 1 y 2
# (commits 0551da4 y 11eafad) editaban ese fichero, así que NUNCA se
# ejecutaron: de ahí que ni el force-reinstall de la headless ni las
# librerías de sistema surtieran efecto.
#
# Estrategia definitiva, con control total y verificación EN BUILD:
#   1) apt-get instala las librerías de sistema (X11/GL) que la
#      variante COMPLETA de OpenCV (que arrastra ultralytics como
#      dependencia) necesita para importar. Así import cv2 funciona
#      SEA CUAL sea la variante que acabe ganando site-packages/cv2/.
#   2) Tras el pip install normal se fuerza de nuevo la variante
#      HEADLESS (defensa independiente: si gana ella, no necesita
#      ninguna librería X11).
#   3) PASO CLAVE: se verifica `import cv2` DURANTE EL BUILD. Si algo
#      estuviera roto, el deploy falla ahí mismo con el error real y
#      visible en los logs de build, no en un bucle de crashes en
#      producción.
# ═══════════════════════════════════════════════════════════════

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Set clásico de librerías que la variante GUI de OpenCV pide al
# importar (libxcb.so.1 era solo el PRIMER eslabón que faltaba; el
# loader resuelve una cada vez y detrás suelen venir GL/glib/SM/Xext).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Capa de dependencias separada para aprovechar la caché de build de
# Railway: requirements.txt cambia mucho menos veces que el código.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    # Defensa independiente: reponer la variante headless por si la
    # opencv-python GUI (dependencia de ultralytics) la sobrescribió.
    && pip install --no-cache-dir --force-reinstall --no-deps 'opencv-python-headless>=4.8.0' \
    # Verificación EN BUILD: si import cv2 falla aquí, el deploy no
    # llega jamás a producción roto.
    && python -c "import cv2; print('VERIFICACION BUILD: import cv2 OK ->', cv2.__version__)"

COPY . .

# El binario está commiteado con 100755; se reafirma por si alguna capa
# intermedia perdiera el bit de ejecución.
RUN chmod +x engine/stockfish-linux-17.1

# Railway inyecta $PORT en runtime; el formato exec no expande variables,
# así que se pasa por sh -c con 8080 como fallback.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]

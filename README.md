# Chess Rekognition API

El "cerebro" del sistema. Esta API construida con **FastAPI** se encarga del procesamiento de imágenes, la lógica de juego mediante Stockfish, la gestión de usuarios y la retransmisión en tiempo real mediante WebSockets.

**¿Qué es FastAPI?** FastAPI es un framework web Python de alto rendimiento para construir APIs REST, con soporte nativo para operaciones asíncronas (`async/await`) y generación automática de documentación interactiva en formato OpenAPI/Swagger.

Sobre FastAPI se construyen todos los endpoints REST (login, registro, CRUD de partidas, Stockfish, retransmisiones) y también los WebSockets para la retransmisión en tiempo real. Se sirve con **Uvicorn** (servidor ASGI) y está desplegado en **Railway**.

La documentación automática está disponible en:
**https://chess-rekognition-api-production.up.railway.app/docs**

**Bibliografía:**
- Documentación oficial: https://fastapi.tiangolo.com
- Repositorio: https://github.com/fastapi/fastapi
- Tutorial oficial completo: https://fastapi.tiangolo.com/tutorial/

---

## Arquitectura del Sistema

### Patrón de Diseño: Arquitectura en 3 Capas

El backend sigue una **arquitectura en 3 capas** con separación clara de responsabilidades. Cada capa tiene una función específica y solo se comunica con la capa inmediatamente inferior:

```
┌──────────────────────────────────────────────────────────┐
│                     Capa de Presentación                 │
│                    (routers / endpoints)                  │
│  Recibe requests HTTP/WS, ejecuta lógica CRUD directa,  │
│  delega en servicios cuando hay complejidad real         │
├──────────────────────────────────────────────────────────┤
│                     Capa de Servicios                    │
│              (services / lógica de negocio compleja)     │
│  Solo donde hay complejidad real: visión por computador, │
│  clasificador ML, motor Stockfish, retransmisión WS      │
├──────────────────────────────────────────────────────────┤
│                  Capa de Infraestructura (core)          │
│      config · security · database · models (SQLModel)    │
│  Configuración global, JWT/bcrypt, BD y definición de    │
│  modelos de datos (ORM + validación Pydantic unificados) │
└──────────────────────────────────────────────────────────┘
```

### Principio de diseño

La capa de servicios **solo existe donde la lógica no cabe razonablemente en el router**: visión por computador (OpenCV), clasificador ML (TensorFlow/MobileNetV2), motor de ajedrez (Stockfish subprocess UCI) y retransmisión en tiempo real (WebSocket broadcast). Para operaciones CRUD sencillas (auth, usuarios, partidas), el propio router gestiona la lógica directamente contra la base de datos, evitando capas intermedias innecesarias.

### Patrones Implementados

| Patrón | Ubicación | Propósito |
|--------|-----------|-----------|
| **Singleton** | `core/config.py` — `get_settings()` con `@lru_cache` | Una única instancia de configuración global |
| **Inyección de Dependencias** | `core/dependencies.py` — `Depends()` de FastAPI | Inyecta sesiones BD, usuario autenticado, etc. |
| **Observer (WebSocket)** | `routers/retransmision.py` — `ConnectionManager` | Broadcast de estado del tablero a múltiples viewers |
| **Callback (Hook)** | `services/training.py` — `_WSCallback` | Keras callback para notificar progreso vía WebSocket |
| **Facade** | `services/vision.py` — `VisionService.detect_and_rectify()` | Orquesta todo el pipeline de visión en un solo método público |
| **Data Transfer Object (DTO)** | `core/models.py` — SQLAlchemy + Pydantic | ORM y schemas de validación unificados en un único módulo |

---

### Flujo de una Petición Típica

```
Cliente (React)                    FastAPI (Python)
     │                                  │
     │  POST /vision/classify           │
     │  (multipart: imagen)             │
     │ ──────────────────────────────►  │
     │                                  │──► router/vision.py
     │                                  │      │
     │                                  │      ├─► services/vision.py
     │                                  │      │    ├─ _detectar_tablero()    [OpenCV]
     │                                  │      │    ├─ _calcular_esquinas()   [vectores]
     │                                  │      │    └─ _rectificar()          [homografía]
     │                                  │      │
     │                                  │      ├─► services/classifier.py
     │                                  │      │    └─ ChessClassifier       [TensorFlow]
     │                                  │      │
     │                                  │      └─► services/vision.py
     │                                  │           └─ board_state_to_fen_board()
     │                                  │
     │  { success, board_state, fen }   │
     │ ◄────────────────────────────── │
```

### Autenticación (JWT Dual Token)

```
┌─────────┐         ┌──────────────┐         ┌───────────┐
│  Login  │ ──────► │  POST /auth  │ ──────► │ Generar   │
│ (creds) │         │  /login      │         │ JWT dual  │
└─────────┘         └──────────────┘         └─────┬─────┘
                                                   │
                            ┌──────────────────────┤
                            │                      │
                     ┌──────▼──────┐        ┌──────▼──────┐
                     │ Access Token │        │Refresh Token│
                     │  30 min     │        │  7 días     │
                     │  en memoria │        │ localStorage│
                     └──────┬──────┘        └──────┬──────┘
                            │                      │
                     ┌──────▼──────┐               │
                     │  Endpoint   │               │
                     │  protegido  │               │
                     │  con JWT    │               │
                     └─────────────┘               │
                            │                      │
                     Si 401 ─────────────────────► │
                                                  │
                                        ┌─────────▼─────────┐
                                        │ POST /auth/refresh │
                                        └───────────────────┘
```

---

## Módulos del Sistema

### 🧠 Visión por Computador (`services/vision.py`)

Pipeline completo de detección y rectificación del tablero:

1. **`_detectar_tablero()`** — OpenCV `findChessboardCornersSB` detecta las 49 intersecciones internas (7×7). Fallback al método clásico. Refinamiento subpíxel.
2. **`_calcular_esquinas_exteriores()`** — Extrapolación vectorial con MARGIN 1.12 para obtener el perímetro real del tablero desde las esquinas internas.
3. **`_rectificar()`** — Homografía (`getPerspectiveTransform` + `warpPerspective`) → vista cenital 400×400.
4. **`_analizar_casillas()`** — Autocalibración dinámica de umbrales por percentiles. Clasifica ocupada/vacía con STD + detección de bordes Canny.
5. **`_calibrar_umbrales()`** — Estadísticos robustos que se adaptan a cualquier iluminación sin calibración manual.

### 🏷️ Clasificador ML (`services/classifier.py`)

- **Modelo**: MobileNetV2 (transfer learning desde ImageNet)
- **Input**: 96×96 píxeles, batch de 64 casillas
- **Clases**: 13 (`empty`, `w_P`…`b_K`)
- **Thread-safe**: `RLock` para evitar condiciones de carrera durante recarga en caliente
- **Cache**: El modelo se mantiene en memoria y se recarga bajo demanda (`/classify/reload`)

### ♟️ Motor de Ajedrez (`services/engine.py`)

- **Stockfish 17.1** vía subprocess con protocolo UCI
- **ELO ajustable** (1320–3190) mediante interpolación lineal a Skill Level
- **Timeout** de 20 segundos con kill forzoso
- **Parseo**: Extrae `bestmove`, score (cp/mate), depth, nodes y PV de la salida UCI

### 🔄 Detector de Movimientos (`services/move_detector.py`)

Compara el estado ML vs FEN anterior:

1. Clasifica el tablero actual con ML
2. Convierte a objeto `chess.Board` (python-chess)
3. Itera sobre movimientos legales del FEN anterior
4. Encuentra el movimiento que produce matching perfecto de piezas
5. Clasifica tipo: `normal`, `capture`, `castling_short/long`, `en_passant`, `promotion`

### 📡 Retransmisión en Tiempo Real (`routers/retransmision.py`)

- **Host** → WebSocket emisor: envía estado del tablero → broadcast a viewers
- **Viewer** → WebSocket receptor: recibe actualizaciones en vivo
- **ConnectionManager**: Mantiene diccionarios de hosts, viewers y último estado (para inyectar a nuevos viewers)
- **Heartbeat**: Los viewers envían "ping" para mantener la conexión activa

### 🎯 Entrenamiento (`services/training.py` + `routers/dataset.py`)

- **Captura y dataset** (`routers/dataset.py`): extrae 64 casillas, sugiere etiqueta por heurística STD, guarda con UUID por clase
- **Training pipeline** (`services/training.py`): `image_dataset_from_directory` → MobileNetV2 + Data Augmentation → EarlyStopping → ReduceLROnPlateau
- **WebSocket broadcasting**: `_WSCallback` (Keras callback) notifica métricas época a época desde el hilo de entrenamiento al event loop principal

---

## Estructura del Proyecto

```
api/
├── main.py                 # Entrypoint FastAPI, CORS, routers, Swagger
├── Procfile                # Comando Railway (uvicorn)
├── requirements.txt        # Dependencias con versiones fijas
│
├── core/                   # Capa de infraestructura
│   ├── config.py           # Settings singleton (@lru_cache), constantes globales
│   ├── security.py         # JWT (access/refresh, 30min/7días) + bcrypt
│   ├── dependencies.py     # get_current_user, get_db — inyección de dependencias
│   ├── database.py         # Engine SQLAlchemy, SessionLocal, Base, get_db
│   └── models.py           # ORM (Usuario, Partida, Retransmision) +
│                           # Schemas Pydantic (validación entrada/salida)
│                           # unificados en un solo módulo
│
├── routers/                # Capa de presentación (endpoints HTTP + WebSocket)
│   ├── auth.py             # POST /login, POST /refresh, GET /whoami
│   ├── usuarios.py         # POST /register (bcrypt + email bienvenida inline)
│   ├── partidas.py         # CRUD /partidas/ con filtrado PI/PR (inline)
│   ├── engine.py           # GET /status, POST /move → delega en services/engine
│   ├── vision.py           # POST /recognize-board, /classify, /detect-move
│   ├── retransmision.py    # WS host/viewer, PATCH, POST host
│   └── dataset.py          # Capture, save, stats, train, WS progreso
│
├── services/               # Capa de lógica compleja
│   ├── vision.py           # Pipeline OpenCV + helpers chess (label_to_piece, board_state_to_fen_board)
│   ├── classifier.py       # ChessClassifier: MobileNetV2, thread-safe, recarga caliente
│   ├── move_detector.py    # Comparación ML vs FEN → movimiento legal python-chess
│   ├── engine.py           # StockfishService: subprocess UCI, ELO, timeout
│   ├── training.py         # Pipeline entrenamiento MobileNetV2 + broadcast WS
│   └── email.py            # Resend API: email de bienvenida asíncrono
│
└── static/                 # Favicon, dataset.html, opencv.html
```

> **Nota de transparencia al frontend:** esta reorganización conceptual no modifica ningún endpoint ni contrato de la API. Las rutas, métodos HTTP, parámetros y respuestas son idénticos. El frontend no requiere ningún cambio.

---

## Endpoints de la API

### Autenticación
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/auth/login` | ❌ | Iniciar sesión (devuelve access + refresh JWT) |
| POST | `/auth/refresh` | ❌ | Renovar access token mediante refresh token |
| GET | `/auth/whoami` | ✅ | Datos del usuario autenticado |

### Usuarios
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/usuarios/register` | ❌ | Registrar nuevo usuario |

### Partidas
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/partidas/` | ✅ | Guardar nueva partida |
| GET | `/partidas/` | ✅ | Listar partidas (filtro opcional PI/PR) |
| GET | `/partidas/{id}` | ✅ | Obtener detalle de partida |
| PATCH | `/partidas/{id}` | ✅ | Actualizar partida |
| DELETE | `/partidas/{id}` | ✅ | Eliminar partida |

### Motor (Stockfish)
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| GET | `/engine/status` | ❌ | Estado del binario Stockfish |
| POST | `/engine/move` | ❌ | Mejor jugada desde posición FEN |

### Visión por Computador
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/vision/recognize-board` | ❌ | Detectar y rectificar tablero |
| POST | `/vision/classify` | ❌ | Clasificar 64 casillas + FEN |
| POST | `/vision/classify/reload` | ❌ | Recargar modelo ML en caliente |
| POST | `/vision/detect-move` | ❌ | Detectar movimiento vs FEN previo |
| GET | `/vision/status` | ❌ | Estado del módulo OpenCV |

### Retransmisión en Tiempo Real
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/retransmision/host` | ✅ | Inicializar retransmisión |
| GET | `/retransmision/status/{token}` | ❌ | Estado de retransmisión |
| PATCH | `/retransmision/{id}` | ✅ | Actualizar retransmisión |
| WS | `/retransmision/ws/host/{token}` | ❌ | WebSocket emisor |
| WS | `/retransmision/ws/viewer/{token}` | ❌ | WebSocket espectador |

### Dataset y Entrenamiento
| Método | Ruta | Protegido | Descripción |
|--------|------|-----------|-------------|
| POST | `/dataset/capture` | ✅ | Capturar 64 casillas desde imagen |
| POST | `/dataset/save` | ✅ | Guardar recortes etiquetados |
| GET | `/dataset/stats` | ✅ | Estadísticas del dataset |
| POST | `/dataset/train` | ✅ | Iniciar entrenamiento del modelo |
| GET | `/dataset/train/status` | ✅ | Estado del entrenamiento (polling) |
| WS | `/dataset/train/ws` | ✅ | Estado del entrenamiento (WebSocket) |

---

## Flujo de Datos: Reconocimiento de Tablero

```
Imagen (cámara / archivo)
        │
        ▼
┌───────────────────────────────┐
│  _detectar_tablero()         │
│  findChessboardCornersSB     │
│  → 49 puntos internos (7×7)  │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  _calcular_esquinas_ext()    │
│  Extrapolación vectorial     │
│  → 4 esquinas exteriores     │
└───────────┬───────────────────┘
            │
            ▼
┌───────────────────────────────┐
│  _rectificar() (homografía)  │
│  400×400, cada casilla 50×50 │
└───────────┬───────────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐  ┌──────────┐
│ Análisis │  │  ML      │
│ clásico  │  │ clasif.  │
│ ocupada/ │  │ 13 clases│
│ vacía    │  │ + FEN    │
└──────────┘  └──────────┘
```

---

## Configuración y Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `DB_HOST` | Host MySQL | — |
| `DB_PORT` | Puerto MySQL | `3306` |
| `DB_USER` | Usuario BD | — |
| `DB_PASSWORD` | Contraseña BD | — |
| `DB_NAME` | Nombre BD | — |
| `JWT_SECRET_KEY` | Clave secreta JWT | — |
| `JWT_ALGORITHM` | Algoritmo JWT | `HS256` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Expiración access token | `30` |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Expiración refresh token | `7` |
| `RESEND_API_KEY` | API Key de Resend | — |
| `RESEND_FROM` | Email remitente | `onboarding@resend.dev` |
| `DATASET_DIR` | Directorio del dataset | `./data/dataset` |
| `MODELS_DIR` | Directorio de modelos | `./data/models` |

---

## Instalación y Ejecución Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno (.env)
# Ver tabla de configuración arriba

# 3. Ejecutar servidor de desarrollo
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Documentación Interactiva

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Despliegue (Railway)

La API está preparada para desplegarse en Railway mediante el `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Las variables de entorno se configuran desde el panel de Railway. El binario de Stockfish (Linux, ~79 MB) se incluye en el repositorio para su uso en producción.

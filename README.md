# Chess Rekognition API

Bienvenido/a a la documentación de la API de Chess Rekognition.

Esta API construida con **FastAPI** se encarga de:

- Procesamiento de las imágenes del tablero de ajedrez.
- Reconocimiento y corrección de la homografía del tablero de ajedrez.
- Recorte y clasificación de las casillas del tablero de ajedrez corregido en vista cenital.
- Entrenamiento de modelo CNN (MobileNetV2 de TensorFlow).
- Reconocimiento de las piezas.
- Lógica de juego mediante Stockfish.
- Gestión de usuarios.
- Guardado de partidas en PGN.
- Retransmisión en tiempo real mediante WebSockets.

**¿Qué es FastAPI?** FastAPI es un framework web Python de alto rendimiento para construir APIs REST, con soporte nativo para operaciones asíncronas (`async/await`) y generación automática de documentación interactiva en formato OpenAPI/Swagger.

Sobre FastAPI se construyen todos los endpoints REST (login, registro, CRUD de partidas, Stockfish, retransmisiones) y también los WebSockets para la retransmisión en tiempo real. Se sirve con **Uvicorn** (servidor ASGI) y está desplegado en **Railway**.

La documentación automática está disponible en:
**https://chess-rekognition-api-production.up.railway.app/docs**

**Bibliografía:**
- Documentación oficial: https://fastapi.tiangolo.com
- Repositorio: https://github.com/fastapi/fastapi
- Tutorial oficial completo: https://fastapi.tiangolo.com/tutorial/

---

## Estructura de archivos

```
api/
├── main.py                 # Punto de entrada de FastAPI
├── Procfile                # Comando de inicio para el despliegue automático en Railway
├── requirements.txt        # Dependencias de Python necesarias para la API
│
├── core/                   # Capa CORE
│   ├── config.py           # Configuración del entorno (.env) con patrón Singleton (@lru_cache)
│   ├── security.py         # Hasheo de contraseñas con bcrypt y utilidades criptográficas de JWT
│   ├── dependencies.py     # Inyectores de dependencia (Depends) de sesión DB y usuario actual
│   ├── database.py         # Configuración de base de datos MySQL e inicialización del motor ORM
│   └── models.py           # Definición ORM (SQLAlchemy) y esquemas de validación (Pydantic DTOs)
│
├── routers/                # Capa ROUTERS
│   ├── auth.py             # Rutas de login, refresh de sesión y whoami
│   ├── usuarios.py         # Registro de usuarios
│   ├── partidas.py         # CRUD de partidas PGN
│   ├── engine.py           # Motor StockFish v17.1
│   ├── vision.py           # Endpoints de diagnóstico OpenCV, clasificación y detección de jugadas
│   ├── retransmision.py    # Gestión de salas activas y WebSockets de emisores y espectadores
│   └── dataset.py          # Captura de crops de casillas, estadísticas y disparadores de entrenamiento
│
├── services/               # Capa SERVICES
│   ├── vision.py           # Servicio para la rectificación de la homografía con OpenCV para la detección y rectificación del tablero de ajedrez.
│   ├── classifier.py       # Servicio de clasificación de piezas de ajedrez mediante un modelo TensorFlow (MobileNetV2).
│   ├── move_detector.py    # Servicio de detección de movimientos de ajedrez mediante comparación visual.
│   ├── engine.py           # Servicio de integración con el motor de ajedrez Stockfish v17.1.
│   ├── training.py         # Servicio de entrenamiento del modelo MobileNetV2.
│   └── email.py            # Servicio de envío de emails transaccionales mediante la API de Resend (https://resend.com).
│
└── static/
    ├── favicon.ico         # Icono de la API
    ├── dataset.html        # Pagina HTML de captura de crops, clasificación y entrenamiento del modelo CNN (MobileNetV2 de TensorFlow)
    ├── opencv.html         # Pagina HTML de pruebas OpenCV (Rectificación, homografía del tablero y valores de Desviación Estándar STD)
    └── reconocimiento.html # Pagina HTML de reconocimiento del tablero y las piezas de ajedrez
```

---

## Arquitectura del Sistema

### Patrón de Diseño: Arquitectura en 3 Capas

El backend sigue una **arquitectura en 3 capas** con separación clara de responsabilidades.

![Arquitectura de 3 Capas](docs/images/arquitectura.png)

Cada capa tiene una función específica y solo se comunica con la capa inmediatamente inferior:

#### Capa de Presentación

Esta capa es dirigida por el archivo main.py e incluye: 

- Endpoints auxiliares.
- Routers registrados con los endpoints.
- Archivos estáticos.

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/` | **GET** | Estado de la API |  |
| `/favicon.ico` | **GET** | Favicon | Sirve static/favicon.ico |
| `/opencv` | **GET** | Pagina HTML de pruebas OpenCV | Sirve static/opencv.html |
| `/dataset` | **GET** | Pagina HTML de captura de crops, clasificación y entrenamiento del modelo CNN (MobileNetV2 de TensorFlow) | Sirve static/dataset.html |
| `/reconocimiento` | **GET** | Pagina HTML de reconocimiento del tablero y las piezas de ajedrez | Sirve static/reconocimiento.html |
| `/docs` | **GET** | Swagger UI personalizado | Usa get_swagger_ui_html() |
| `CORSMiddleware` | — | Permite peticiones cross-origin desde cualquier origen | `allow_origins=['*']`, methods: GET/POST/PUT/DELETE/OPTIONS/PATCH, credentials=True |
| `include_router(x7)` | — | auth, usuarios, partidas, engine, vision, dataset, retransmision | app.include_router() para cada modulo del directorio routers/ |

#### Capa Routers

Esta capa contiene los diferentes endpoints y websockets registrados en la capa de presentación (main.py). 

Los diferentes routers con los que cuenta la API y sus correspondientes endpoints son:

##### auth.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/auth/login` | **POST** | Autentica con username + password y devuelve access token (30 min) y refresh token (7 días). |
| `/auth/refresh` | **POST** | Emite un nuevo access token a partir de un refresh token válido. |
| `/auth/whoami` | **GET** | Devuelve los datos del usuario identificado por el access token. |

##### usuarios.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/usuarios/register` | **POST** | Registra un nuevo usuario y envía un email de bienvenida via Resend. |

##### partidas.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/partidas/` | **POST** | Crea una nueva partida asociada al usuario autenticado. |
| `/partidas/` | **GET** | Lista todas las partidas del usuario autenticado, con filtro opcional por tipo (PI: Partida Introducida / PR: Partida Retransmitida). |
| `/partidas/{id_partida}` | **GET** | Obtiene el detalle de una partida por ID. Solo devuelve la partida si pertenece al usuario. |
| `/partidas/{id_partida}` | **PATCH** | Actualiza los campos enviados de una partida del usuario autenticado. |
| `/partidas/{id_partida}` | **DELETE** | Elimina una partida del usuario autenticado. (Actualmente no se usa en la app). |

##### engine.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/engine/status` | **GET** | Verifica el estado y la versión del motor Stockfish v17.1. |
| `/engine/move` | **POST** | Obtiene la mejor jugada dada una posición en FEN. Parámetros opcionales: elo (1320-3190) y depth (1-30). |

##### vision.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/vision/status` | **GET** | Devuelve el estado operativo del módulo y la versión de OpenCV. |
| `/vision/recognize-board`| **POST** | Detecta y rectifica el tablero aplicando la homografía. Devuelve vista cenital 400x400. |
| `/vision/classify` | **POST** | Clasifica las 64 casillas del tablero con MobileNetV2 y genera el FEN completo. |
| `/vision/classify/reload` | **POST** | Recarga el modelo ML desde disco sin reiniciar el servidor. |
| `/vision/detect-move` | **POST** | Compara el estado actual del tablero con un FEN previo y detecta el movimiento realizado. |

##### retransmision.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/retransmision/host` | **POST** | Crea una nueva retransmisión con token único y la marca como activa. |
| `/retransmision/status/{token}`| **GET** | Devuelve si la retransmisión está activa y cuántos viewers hay conectados. |
| `/retransmision/{id_retransmision}`| **PATCH**| Actualiza los metadatos de una retransmisión del usuario autenticado. |
| `/retransmision/ws/host/{token}`| **WS** | WebSocket del emisor (host): recibe el estado del tablero y hace broadcast a los viewers. |
| `/retransmision/ws/viewer/{token}`| **WS** | WebSocket del espectador (viewer): recibe actualizaciones en tiempo real del tablero. |

##### dataset.py

| Endpoint / Componente | Método | Detalle |
| :--- | :---: | :--- |
| `/dataset/capture` | **POST** | Detecta el tablero, rectifica la homografía y devuelve las 64 casillas recortadas (crops) con etiquetado sugerido por el STD. |
| `/dataset/save` | **POST** | Recibe las casillas (crops) etiquetadas y las guarda en el directorio de su clase con nombre UUID único. |
| `/dataset/stats` | **GET** | Devuelve el número de imágenes por clase y el total acumulado. |
| `/dataset/train` | **POST** | Lanza el entrenamiento de MobileNetV2 en un hilo secundario que generar el archivo del modelo. |
| `/dataset/train/status` | **GET** | Devuelve el estado actual del entrenamiento (polling). |
| `/dataset/train/ws` | **WS** | WebSocket de progreso del entrenamiento en tiempo real para consola. El token JWT se pasa como query param: ?token=<access_token> |

#### Capa Services

Esta capa contiene los diferentes servicios asíncronos y síncronos con los que cuenta la API para procesar lógica de negocio compleja.

#### Capa Core

La capa core representa el núcleo de la infraestructura del sistema, gestionando la base de datos, la configuración global y las herramientas transversales de seguridad.

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

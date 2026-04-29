# chess-rekognition-api

FastAPI - Chess Rekognition.

## Arquitectura
```text
api/
├── main.py                  # Router principal + config Swagger
├── Procfile                 # Comando de arranque para Railway
├── requirements.txt         # Dependencias
├── .env                     # Variables de entorno (JWT secret, DB, Resend). No se sube a GitHub y se configura en Railway en Variables
│
├── static/
│   └── favicon.ico          # Favicon de la API
│
├── core/
│   ├── __init__.py
│   ├── config.py            # Settings (carga .env)
│   ├── security.py          # Lógica JWT (crear/verificar tokens)
│   └── dependencies.py      # Dependencias reutilizables (get_current_user)
│
├── db/
│   ├── __init__.py
│   └── database.py          # Conexión SQLAlchemy
│
├── models/
│   ├── __init__.py
│   └── usuarios.py          # Modelo ORM tabla usuarios
│
├── schemas/
│   ├── __init__.py
│   └── usuarios.py          # Pydantic schemas (validación + Swagger docs)
│
├── services/
│   ├── __init__.py
│   ├── auth.py              # Lógica de negocio: login, tokens, whoami
│   ├── usuarios.py          # Lógica: registro + envío de correo
│   ├── email.py             # Servicio de envío de emails (Resend)
│   └── vision.py            # Lógica OpenCV: detección y rectificación de tablero
│
└── routers/
    ├── __init__.py
    ├── auth.py              # Endpoints: /login /refresh /whoami
    ├── usuarios.py          # Endpoints: /register
    ├── vision.py            # Endpoints: /vision (reconocimiento de imagen)
    └── retransmision.py     # Endpoints: /retransmision (WebSockets para emisión en directo)
```

## Ejecución en Local

Para levantar el backend en tu entorno local, sigue estos pasos:

1. **Crear el entorno virtual:**
   ```bash
   python -m venv venv
   ```

2. **Activar el entorno virtual:**
   - En **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```
   - En **macOS / Linux**:
     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Variables de Entorno:**
   Asegúrate de crear un archivo `.env` en la raíz de la carpeta `api/` con las credenciales necesarias, por ejemplo:
   ```env
   DATABASE_URL=postgresql://...
   JWT_SECRET_KEY=tu_secreto_jwt
   RESEND_API_KEY=re_...
   ```

5. **Iniciar el servidor:**
   Ejecuta FastAPI en modo recarga automática para desarrollo:
   ```bash
   uvicorn main:app --reload
   ```
   *La API estará accesible localmente en `http://localhost:8000`.*

## Documentación

Swagger UI: https://chess-rekognition-api-production.up.railway.app/docs
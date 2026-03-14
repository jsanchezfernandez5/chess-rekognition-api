# chess-rekognition-api
FastAPI - Chess Rekognition.

# Arquitectura del API
api/
├── main.py                  # Router principal + config Swagger
├── requirements.txt         # Dependencias
├── .env                     # Variables de entorno (JWT secret, DB, Resend)
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
│   ├── usuarios.py          # Modelo ORM tabla usuarios
│
├── schemas/
│   ├── __init__.py
│   ├── usuarios.py           # Pydantic schemas (validación + Swagger docs)
│
├── services/
│   ├── __init__.py
│   ├── auth.py              # Lógica de negocio: login, tokens, whoami
│   ├── usuarios.py          # Lógica: registro + envío de correo
│   ├── email.py
│
└── routers/
    ├── __init__.py
    ├── auth.py              # Endpoints: /login /refresh /whoami
    ├── usuarios.py          # Endpoints: /register
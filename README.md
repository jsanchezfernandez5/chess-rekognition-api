# Chess Rekognition API

El "cerebro" del sistema. Esta API construida con **FastAPI** se encarga del procesamiento de imágenes, la lógica de juego mediante Stockfish, la gestión de usuarios y la retransmisión en tiempo real mediante WebSockets.

---

## Infraestructura y Servicios
Para garantizar un rendimiento óptimo y alta disponibilidad, la API utiliza:

*   **Railway**: Hosting del servidor de aplicaciones (FastAPI) y la base de datos gestionada **MySQL**. 
    *   Configuración automática mediante `Procfile`.
    *   Persistencia de datos garantizada en el cloud.
*   **Resend**: Integración para el envío de correos electrónicos transaccionales (como el registro de usuarios).
*   **Stockfish 17.1**: Motor de ajedrez integrado para el análisis y juego contra la IA.

---

## Stack Técnico y Librerías

El backend está desarrollado en **Python 3.11+** utilizando las siguientes bibliotecas clave:

*   **FastAPI**: Framework moderno y rápido para construir APIs con WebSockets.
*   **SQLAlchemy**: ORM para la comunicación fluida con MySQL.
*   **PyMySQL**: Driver de conexión para la base de datos.
*   **OpenCV (`opencv-python`)**: Procesamiento de imagen para visión artificial.
*   **PyJWT**: Generación y validación de tokens de seguridad (JWT).
*   **Passlib (Bcrypt)**: Encriptación segura de contraseñas.
*   **Pydantic**: Validación de datos y esquemas de entrada/salida.
*   **Resend Python SDK**: Comunicación con el servicio de correo.
*   **Python-Multipart**: Para la recepción y gestión de archivos/imágenes.

---

## Estructura del Proyecto

```text
api/
├── main.py              # Punto de entrada y configuración de Swagger
├── Procfile             # Instrucciones de ejecución para Railway
├── requirements.txt     # Listado de librerías y versiones
├── .env                 # Configuración sensible (API Keys, DB URL)
├── core/                # Configuración global y seguridad (JWT)
├── db/                  # Configuración de base de datos y sesión
├── models/              # Modelos ORM (SQLAlchemy)
├── schemas/             # Modelos de validación (Pydantic)
├── routers/             # Endpoints de la API organizados por módulos
├── services/            # Lógica de negocio (Email, Motor, Visión)
└── static/              # Archivos estáticos y recursos
```

---

## Instalación Local

1.  **Instalar dependencias**:
    Es necesario instalar todas las librerías indicadas en el archivo de requisitos:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configurar Variables de Entorno**:
    Crea un archivo `.env` basado en la configuración de producción:
    ```env
    DATABASE_URL=mysql+pymysql://user:password@host:port/dbname
    JWT_SECRET=tu_secreto_para_tokens
    RESEND_API_KEY=re_xxxxxxxxxxxx
    MAIL_FROM=tu_email_configurado@dominio.com
    ```

3.  **Ejecutar Servidor**:
    ```bash
    uvicorn main:app --reload
    ```

---

## Documentación Interactiva
Una vez arrancado el servidor, puedes acceder a la documentación interactiva autogenerada en:
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
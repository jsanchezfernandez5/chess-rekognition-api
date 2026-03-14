# services/email.py
# Servicio de email usando Resend (https://resend.com).
import asyncio
import resend
from core.config import settings

# API RESEND
resend.api_key = settings.RESEND_API_KEY

# Función para enviar un correo de bienvenida al usuario recién registrado.
async def send_welcome_email(nombre: str, mail: str) -> None:
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
        <h2 style="color: #2c3e50;">¡Bienvenido a Chess Rekognition, {nombre}!</h2>
        <p>Tu cuenta ha sido creada correctamente. Ya puedes iniciar sesión y comenzar a registrar tus partidas.</p>
      </body>
    </html>
    """
    # asyncio.to_thread ejecuta la función síncrona en un thread del pool,
    # liberando el event loop mientras espera la respuesta HTTP de Resend.
    await asyncio.to_thread(
        resend.Emails.send,
        {
            "from": settings.RESEND_FROM,
            "to": mail,
            "subject": "Bienvenido a Chess Rekognition",
            "html": html_body,
        }
    )

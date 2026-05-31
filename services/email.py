"""
Servicio de envío de emails transaccionales mediante la API de Resend (https://resend.com).

Actualmente se usa para enviar el email de bienvenida tras el registro en POST /usuarios/register.
"""
import asyncio
import resend
from core.config import settings

# API RESEND
resend.api_key = settings.RESEND_API_KEY

# Función para enviar un correo de bienvenida al usuario recién registrado.
# Se llama desde el endpoint POST /register.
async def send_welcome_email(nombre: str, mail: str) -> None:
    """
    Envía un correo de bienvenida al usuario tras completar su registro.

    Args:
        nombre: Nombre del usuario destinatario del correo.
        mail: Dirección de correo electrónico del destinatario.

    Raises:
        Exception: Si la API de Resend devuelve un error. La excepción se propaga
                   para que el llamador pueda manejar el fallo adecuadamente.
    """
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
        <h2 style="color: #2c3e50;">¡Bienvenido a Chess Rekognition, {nombre}!</h2>
        <p>Tu cuenta ha sido creada correctamente. Ya puedes iniciar sesión y comenzar a registrar tus partidas.</p>
      </body>
    </html>
    """
    
    # Ejecuta la función síncrona en un thread del pool, liberando el event loop
    await asyncio.to_thread(
        resend.Emails.send,
        {
            "from": settings.RESEND_FROM,
            "to": mail,
            "subject": "Bienvenido a Chess Rekognition",
            "html": html_body,
        }
    )

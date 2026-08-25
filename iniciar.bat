@echo off
chcp 65001 >nul
title Chess Rekognition API - Local

echo ============================================
echo   Chess Rekognition API - Arranque Local
echo ============================================
echo.

:: Verifica que exista el archivo .env
if not exist ".env" (
    echo [ERROR] No se encuentra el archivo .env
    echo Crea un archivo .env a partir de .env.example con tus variables de entorno:
    echo   copy .env.example .env
    echo.
    pause
    exit /b 1
)

:: Verifica que exista el entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual. Verifica que Python este instalado.
        pause
        exit /b 1
    )
    echo [OK] Entorno virtual creado.
    echo.
    echo [INFO] Instalando dependencias (esto puede tardar varios minutos)...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Fallo la instalacion de dependencias.
        pause
        exit /b 1
    )
    echo [OK] Dependencias instaladas.
    echo.
)

:: Activa el entorno virtual e inicia el servidor
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [INFO] Iniciando servidor FastAPI en http://localhost:8000
echo [INFO] Documentacion Swagger: http://localhost:8000/docs
echo [INFO] Para detener el servidor, presiona Ctrl+C
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Inicia el backend en el puerto 8081 (mismo que espera el frontend).
Set-Location $PSScriptRoot
& .\venv\Scripts\uvicorn.exe main:app --reload --host 0.0.0.0 --port 8081

"""Punto de entrada alternativo — delega en main.py para mantener una sola app."""

from main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=True)

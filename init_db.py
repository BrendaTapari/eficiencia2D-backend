"""Crea las tablas en PostgreSQL. Ejecutar desde la raíz del proyecto."""
from database import init_db

if __name__ == "__main__":
    init_db()
    print("Tablas creadas correctamente.")

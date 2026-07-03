import os
import logging
from typing import Generator

import jwt
from fastapi import Header, HTTPException
from sqlalchemy.orm import Session

from database.database import SessionLocal

log = logging.getLogger(__name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_USER_CLAIM = os.environ.get("JWT_USER_CLAIM", "sub")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(authorization: str = Header(...)) -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET no configurado")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Token inválido o ausente")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token inválido")

    user_id = payload.get(JWT_USER_CLAIM)
    if not user_id:
        raise HTTPException(status_code=401, detail="Token sin identificador de usuario")

    return str(user_id)

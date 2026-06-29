import logging
import mimetypes
import os
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

import boto3
from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_URL_EXPIRATION_SECONDS = 3600


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} no está definida. Configurala en el archivo .env del proyecto.")
    return value


def _get_bucket_name() -> str:
    return _require_env("R2_BUCKET_NAME")


def _get_r2_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=_require_env("R2_ENDPOINT_URL"),
        aws_access_key_id=_require_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_require_env("R2_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("R2_REGION", "auto"),
    )


def construir_ruta_archivo(usuario_id: UUID | str, proyecto_id: UUID | str, nombre_archivo: str) -> str:
    """Devuelve la clave del objeto en R2: {usuario_id}/{proyecto_id}/{nombre_archivo}."""
    nombre_seguro = Path(nombre_archivo).name
    if not nombre_seguro:
        raise ValueError("nombre_archivo no puede estar vacío")
    return f"{usuario_id}/{proyecto_id}/{nombre_seguro}"


def subir_archivo(
    archivo: bytes | BinaryIO,
    usuario_id: UUID | str,
    proyecto_id: UUID | str,
    nombre_archivo: str,
    *,
    content_type: str | None = None,
) -> str:
    """
    Sube un archivo a Cloudflare R2 y devuelve la ruta (clave del objeto).

    Estructura virtual: {usuario_id}/{proyecto_id}/{nombre_archivo}
    """
    ruta = construir_ruta_archivo(usuario_id, proyecto_id, nombre_archivo)

    if content_type is None:
        content_type, _ = mimetypes.guess_type(nombre_archivo)
    extra_args: dict[str, str] = {}
    if content_type:
        extra_args["ContentType"] = content_type

    client = _get_r2_client()
    bucket = _get_bucket_name()

    try:
        if isinstance(archivo, bytes):
            client.put_object(Bucket=bucket, Key=ruta, Body=archivo, **extra_args)
        else:
            client.upload_fileobj(archivo, bucket, ruta, ExtraArgs=extra_args or None)
    except (BotoCoreError, ClientError) as exc:
        logger.exception("Error al subir archivo a R2: %s", ruta)
        raise RuntimeError("No se pudo subir el archivo a R2") from exc

    logger.info("Archivo subido a R2: %s", ruta)
    return ruta


def obtener_url_archivo(
    ruta: str,
    *,
    expiration_seconds: int | None = None,
) -> str:
    """
    Genera una URL firmada para descargar el archivo desde R2.

    `ruta` debe ser la clave del objeto, por ejemplo:
    {usuario_id}/{proyecto_id}/{nombre_archivo}
    """
    if not ruta or ruta.strip() != ruta or ".." in Path(ruta).parts:
        raise ValueError("ruta de archivo inválida")

    client = _get_r2_client()
    bucket = _get_bucket_name()
    expires_in = expiration_seconds or int(
        os.environ.get("R2_URL_EXPIRATION_SECONDS", DEFAULT_URL_EXPIRATION_SECONDS)
    )

    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": ruta},
            ExpiresIn=expires_in,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.exception("Error al generar URL de descarga para: %s", ruta)
        raise RuntimeError("No se pudo generar la URL del archivo") from exc


def descargar_archivo(ruta: str) -> bytes:
    """Descarga el contenido de un objeto desde R2."""
    if not ruta or ruta.strip() != ruta or ".." in Path(ruta).parts:
        raise ValueError("ruta de archivo inválida")

    client = _get_r2_client()
    bucket = _get_bucket_name()

    try:
        response = client.get_object(Bucket=bucket, Key=ruta)
        return response["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        logger.exception("Error al descargar archivo de R2: %s", ruta)
        raise RuntimeError("No se pudo descargar el archivo de R2") from exc


def eliminar_archivo(ruta: str) -> None:
    """Elimina un objeto de R2."""
    if not ruta or ruta.strip() != ruta or ".." in Path(ruta).parts:
        raise ValueError("ruta de archivo inválida")

    client = _get_r2_client()
    bucket = _get_bucket_name()

    try:
        client.delete_object(Bucket=bucket, Key=ruta)
    except (BotoCoreError, ClientError) as exc:
        logger.exception("Error al eliminar archivo de R2: %s", ruta)
        raise RuntimeError("No se pudo eliminar el archivo de R2") from exc

    logger.info("Archivo eliminado de R2: %s", ruta)

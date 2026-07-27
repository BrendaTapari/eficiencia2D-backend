# Despliegue en VM de Google Cloud (Debian) — Eficiencia2D Backend

> **Aclaración importante sobre el stack.** El backend es **Python + FastAPI**, servido
> con **uvicorn**. No hay Express/Node ni un "motor C++": todo el pipeline de geometría
> (parseo OBJ/STL, clasificación, nesting, DXF y PDF) es Python puro y usa las librerías
> `ezdxf`, `matplotlib`, `trimesh`, `shapely`, `numpy`, `scipy`. Por lo tanto **no hay
> ejecutable que recompilar**: el "build" es crear el entorno virtual e instalar
> `requirements-server.txt`. No existe `package.json` ni script `build:engine` porque no
> aplican a este stack.

## Qué cambió en el código para el deploy

1. **CORS** (`main.py`): los dominios oficiales `https://eficiencia2d.com.ar` y
   `https://www.eficiencia2d.com.ar` ahora se permiten **siempre** (además de lo que
   llegue por variables de entorno). Funciona aunque falten las env vars de CORS.
2. **Binding de IP/puerto** (`main.py`): `python main.py` ahora escucha en `HOST`
   (default `0.0.0.0`, recibe tráfico externo) y `PORT` (default `80`), con `RELOAD`
   opcional. Igual, en producción se recomienda uvicorn por CLI/systemd (abajo).
3. **Rutas de archivos temporales** (`api/routes/uploads.py`): `UPLOAD_DIR`/`CACHE_DIR`
   se anclan a una raíz **absoluta** vía `os.path.join` (Linux-safe), configurable con
   `DATA_DIR`. Antes eran relativas al CWD; un servicio systemd puede arrancar desde
   cualquier carpeta, así que ahora son independientes del directorio actual.

## Paso a paso en la VM (Debian)

```bash
# 1. Dependencias del sistema
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip nginx git

# 2. Traer el código (ajustar la ruta destino)
sudo mkdir -p /opt/eficiencia2D-backend
sudo chown "$USER" /opt/eficiencia2D-backend
git clone <URL_DEL_REPO> /opt/eficiencia2D-backend
cd /opt/eficiencia2D-backend

# 3. "Build" del backend: entorno virtual + dependencias (equivale a build:engine)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-server.txt

# 4. Configurar variables de entorno
cp .env.example .env   # si no hay ejemplo, crearlo a mano (ver lista abajo)
nano .env

# 5. Carpeta de datos temporales (fuera del checkout, persiste entre deploys)
sudo mkdir -p /var/lib/eficiencia2d
sudo chown www-data:www-data /var/lib/eficiencia2d
```

### Variables de entorno mínimas (`.env`)

```
DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME
JWT_SECRET=<secreto largo aleatorio>
JWT_ALGORITHM=HS256
MP_ACCESS_TOKEN=<token de Mercado Pago>
FRONTEND_URL=https://eficiencia2d.com.ar
DATA_DIR=/var/lib/eficiencia2d
# Correo (fastapi-mail): MAIL_USERNAME, MAIL_PASSWORD, MAIL_FROM, MAIL_SERVER, MAIL_PORT
```

> ⚠️ Si alguno de esos secretos se compartió en texto plano en algún chat o commit,
> **rotarlo** antes del deploy (Mercado Pago, JWT_SECRET, credenciales de la base y del
> correo).

## Arranque

### Opción A (recomendada): uvicorn en 127.0.0.1:8081 + nginx en 80/443

```bash
# systemd para el backend — editar el puerto a 8081 en el .service si usás nginx
sudo cp deploy/eficiencia2d.service /etc/systemd/system/eficiencia2d.service
sudo sed -i 's/--port 80/--port 8081/' /etc/systemd/system/eficiencia2d.service
sudo systemctl daemon-reload
sudo systemctl enable --now eficiencia2d

# nginx como reverse proxy (maneja TLS y el body de 500 MB)
sudo cp deploy/nginx-eficiencia2d.conf /etc/nginx/sites-available/eficiencia2d
sudo ln -s /etc/nginx/sites-available/eficiencia2d /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d eficiencia2d.com.ar -d www.eficiencia2d.com.ar
```

### Opción B: uvicorn directo en el puerto 80 (sin nginx)

El `.service` incluido ya bindea `0.0.0.0:80` como `www-data` usando
`AmbientCapabilities=CAP_NET_BIND_SERVICE` (permite el puerto <1024 sin ser root):

```bash
sudo cp deploy/eficiencia2d.service /etc/systemd/system/eficiencia2d.service
sudo systemctl daemon-reload
sudo systemctl enable --now eficiencia2d
```

O, para una prueba rápida a mano:

```bash
sudo -E env PATH="$PATH" ./start-server.sh      # usa PORT=80 por defecto
```

## Firewall de Google Cloud

En la consola de GCP, permitir el tráfico entrante a los puertos correspondientes
(regla de firewall de la VPC), no solo dentro de la VM:

```
# HTTP y HTTPS hacia la VM (ajustar el tag de red de la instancia)
gcloud compute firewall-rules create allow-http  --allow=tcp:80  --direction=INGRESS
gcloud compute firewall-rules create allow-https --allow=tcp:443 --direction=INGRESS
```

Y apuntar los registros DNS `A` de `eficiencia2d.com.ar` y `www` a la IP externa de la VM.

## Verificación

```bash
curl -i http://localhost/           # {"message":"Bienvenido a la API..."}
systemctl status eficiencia2d
journalctl -u eficiencia2d -f       # logs en vivo
```

# 🏗️ Motor de Corte 3D a 2D (Backend)

Este es el backend principal para el procesamiento de modelos 3D arquitectónicos y su conversión a planos 2D optimizados. 

La API está construida para ser rápida y resistente, resolviendo el desafío de manejar archivos pesados (+50MB) en servidores con memoria RAM limitada sin sufrir caídas por *Out of Memory* (OOM) o *Timeouts* HTTP.

## 🚀 Características Principales

* **Recepción Segura de Archivos:** Implementación de *chunking* para guardar archivos temporalmente en disco sin saturar la RAM.
* **Geometría Computacional Avanzada:** Preparado para *unfolding* de caras coplanares y cortes transversales Z para topografías usando `trimesh` y `numpy`.
* **Exportación Vectorial:** Generación de archivos `.dxf` listos para ser interpretados por software de corte láser.
* **Integración Frontend:** CORS configurado dinámicamente mediante variables de entorno para integrarse sin fricción con un cliente en TypeScript.

## 🛠️ Stack Tecnológico

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Servidor ASGI:** Uvicorn
* **Procesamiento 3D:** Trimesh, Shapely, NumPy
* **Exportación 2D:** Ezdxf
* **Base de Datos (Futuro):** SQLAlchemy

## 📂 Estructura del Proyecto

```text
backend_laser/
├── api/
│   └── routes/
│       └── upload.py        # Endpoint de recepción de archivos pesados
├── core/
│   ├── geometry.py          # Lógica matemática de conversión 3D -> 2D
│   └── exporter.py          # Generación de archivos DXF
├── database/                # Configuración de BD y modelos (en desarrollo)
├── temp/
│   ├── uploads/             # Almacenamiento temporal de modelos entrantes (.stl, .obj)
│   └── outputs/             # Archivos procesados listos para descarga (.dxf)
├── .env                     # Variables de entorno
├── main.py                  # Punto de entrada de la API
└── requirements.txt         # Dependencias de Python



## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_PROYECTO>
```

### 2. Crear un entorno virtual

Desde la raíz del proyecto:

```bash
python3 -m venv venv
```

### 3. Activar el entorno virtual

#### Linux / macOS

```bash
source venv/bin/activate
```
#### Windows

```bash
venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Configurar variables de entorno

Crea un archivo llamado `.env` en la raíz del proyecto.

Ejemplo:

```env
FRONTEND_URL=http://localhost:3000
```

> **Nota:** Ajusta la URL según el entorno donde se encuentre ejecutándose tu frontend.

### 6. Iniciar el servidor

```bash
uvicorn main:app --reload
```

---

## Acceso a la aplicación

Una vez iniciado el servidor, los siguientes recursos estarán disponibles:

| Recurso | URL |
|----------|----------|
| API | `http://127.0.0.1:8000` |
| Documentación Swagger | `http://127.0.0.1:8000/docs` |
| Documentación ReDoc | `http://127.0.0.1:8000/redoc` |

---

## Modo desarrollo

El parámetro `--reload` habilita la recarga automática del servidor cada vez que se detectan cambios en el código fuente.

```bash
uvicorn main:app --reload
```

Esto facilita el desarrollo al evitar reiniciar manualmente la aplicación después de cada modificación.

---

## Estructura de configuración

Archivo `.env`:

```env
FRONTEND_URL=http://localhost:3000
```

| Variable | Descripción |
|-----------|-------------|
| `FRONTEND_URL` | URL del frontend autorizada para realizar solicitudes al backend. |

---

## Solución de problemas

### El comando `uvicorn` no se encuentra

Ejecuta:

```bash
pip install uvicorn
```

O verifica que el entorno virtual esté activado.

### Error al instalar dependencias

Actualiza `pip` e intenta nuevamente:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Error de CORS

Verifica que la variable `FRONTEND_URL` coincida con la URL desde la que se está ejecutando el frontend.
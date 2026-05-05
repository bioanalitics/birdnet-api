"""
BirdNET Detection API — backend para identificación automática de aves.
Usa birdnetlib sobre BirdNET-Analyzer.
Desplegado en Render.com (Python 3.11).

CAMBIO CLAVE: el modelo se carga en el evento 'lifespan' (startup asíncrono),
permitiendo que uvicorn abra el puerto ANTES de descargar el modelo.
Esto evita que Render mate el proceso por no detectar el puerto a tiempo.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import traceback
from typing import Optional


# ── Estado global del modelo ───────────────────────────────────────────────────
# Se llena durante el startup, no en tiempo de importación.
MODEL = {"analyzer": None, "listo": False, "error": None}


# ── Lifespan: carga el modelo DESPUÉS de que uvicorn abre el puerto ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Código que corre al iniciar el servidor (antes de recibir requests).
    Al usar lifespan, uvicorn ya tiene el puerto abierto cuando esto corre,
    evitando el timeout de Render.
    """
    print("==> Iniciando servidor BirdNET API...")
    try:
        from birdnetlib.analyzer import Analyzer
        print("==> Descargando/cargando modelo BirdNET (puede tardar 1-2 min)...")
        MODEL["analyzer"] = Analyzer()
        MODEL["listo"]    = True
        print("==> Modelo BirdNET cargado correctamente. Listo para analizar.")
    except Exception as e:
        MODEL["error"] = str(e)
        MODEL["listo"] = False
        print(f"==> ERROR al cargar modelo: {e}")
        traceback.print_exc()
        # No relanzamos — el servidor sigue vivo y devuelve 503 en /analyze

    yield  # <-- el servidor corre aquí

    # Código de cierre (opcional)
    print("==> Apagando servidor BirdNET API.")


# ── Aplicación FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BirdNET Detection API",
    description = "Identificación automática de aves desde audio WAV/MP3.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """
    Verifica estado de la API y del modelo.
    Shiny llama esto al cargar la pestaña para despertar el servidor.
    """
    return {
        "status"       : "ok" if MODEL["listo"] else "iniciando",
        "modelo_listo" : MODEL["listo"],
        "error"        : MODEL["error"],
        "mensaje"      : (
            "API BirdNET activa y lista." if MODEL["listo"]
            else "Modelo cargando, intenta en 30 segundos."
        ),
    }


# ── Análisis de audio ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_audio(
    audio    : UploadFile      = File(...),
    lat      : Optional[float] = Form(None),
    lon      : Optional[float] = Form(None),
    week     : Optional[int]   = Form(None),
    min_conf : float           = Form(0.10),
):
    """
    Analiza un archivo WAV o MP3 y devuelve las especies detectadas.

    Parámetros:
      audio    → archivo WAV o MP3 (obligatorio)
      lat, lon → coordenadas del lugar de grabación (mejoran la precisión)
      week     → semana del año 1-48 (filtra por época del año)
      min_conf → confianza mínima 0-1 (default: 0.10)

    Para Chile: lat entre -17 y -56, lon entre -66 y -75.
    """
    # Verificar que el modelo esté listo
    if not MODEL["listo"]:
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"Modelo aún no disponible: {MODEL['error'] or 'cargando'}. "
                "Espera 30 segundos y reintenta."
            )
        )

    # Validar formato
    extension = os.path.splitext(audio.filename or "audio.wav")[-1].lower()
    if extension not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(
            status_code = 400,
            detail      = f"Formato '{extension}' no soportado. Usa WAV o MP3."
        )

    if not (0.0 <= min_conf <= 1.0):
        raise HTTPException(
            status_code = 400,
            detail      = "min_conf debe estar entre 0 y 1."
        )

    tmp_path = None
    try:
        # Guardar audio en archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            contenido = await audio.read()
            if len(contenido) == 0:
                raise HTTPException(status_code=400, detail="Archivo de audio vacío.")
            tmp.write(contenido)
            tmp_path = tmp.name

        # ── Análisis con birdnetlib ────────────────────────────────────────────
        from birdnetlib import Recording

        recording = Recording(
            MODEL["analyzer"],
            tmp_path,
            lat      = lat,
            lon      = lon,
            week     = week if week else -1,
            min_conf = min_conf,
            overlap  = 0.0,
        )
        recording.analyze()

        detecciones_raw = recording.detections

        # ── Formatear respuesta ────────────────────────────────────────────────
        detecciones = []
        for det in detecciones_raw:
            confianza = round(float(det.get("confidence", 0)), 4)
            detecciones.append({
                "inicio_seg"        : round(float(det.get("start_time", 0)), 2),
                "fin_seg"           : round(float(det.get("end_time",   3)), 2),
                "nombre_cientifico" : det.get("scientific_name", ""),
                "nombre_comun"      : det.get("common_name",     ""),
                "confianza"         : confianza,
                "confianza_pct"     : f"{confianza * 100:.1f}%",
            })

        detecciones.sort(key=lambda x: (x["inicio_seg"], -x["confianza"]))

        # Resumen por especie única
        vistas = {}
        for d in detecciones:
            sp = d["nombre_cientifico"]
            if sp not in vistas:
                vistas[sp] = {
                    "nombre_cientifico" : sp,
                    "nombre_comun"      : d["nombre_comun"],
                    "max_confianza"     : d["confianza"],
                    "max_confianza_pct" : d["confianza_pct"],
                    "n_detecciones"     : 1,
                }
            else:
                vistas[sp]["n_detecciones"] += 1
                if d["confianza"] > vistas[sp]["max_confianza"]:
                    vistas[sp]["max_confianza"]     = d["confianza"]
                    vistas[sp]["max_confianza_pct"] = d["confianza_pct"]

        especies = sorted(vistas.values(), key=lambda x: -x["max_confianza"])

        return {
            "ok"                : True,
            "archivo"           : audio.filename,
            "n_detecciones"     : len(detecciones),
            "n_especies"        : len(especies),
            "filtros_aplicados" : {
                "lat"      : lat,
                "lon"      : lon,
                "week"     : week,
                "min_conf" : min_conf,
            },
            "especies"          : especies,
            "detecciones"       : detecciones,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"[ERROR] Fallo al analizar '{audio.filename}':")
        traceback.print_exc()
        raise HTTPException(
            status_code = 500,
            detail      = f"Error al procesar el audio: {str(e)}"
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

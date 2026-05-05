"""
BirdNET Detection API — backend para identificación automática de aves.
Usa birdnetlib (wrapper sobre BirdNET-Analyzer) para el análisis de audio.
Desplegado en Render.com (Python 3.11), consumido desde Shiny via httr.

Endpoints:
  GET  /          → health check / despertar servidor
  POST /analyze   → recibe audio WAV/MP3, devuelve especies detectadas
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import tempfile
import os
import traceback
from typing import Optional


# ── Inicialización del analizador (una sola vez al arrancar) ───────────────────
# Esto es crítico: cargarlo aquí evita hacerlo en cada petición.
# El modelo TFLite se descarga automáticamente si no existe en disco.
print("Cargando modelo BirdNET...")
try:
    analyzer = Analyzer()
    print("Modelo BirdNET cargado correctamente.")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    analyzer = None


# ── Aplicación FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BirdNET Detection API",
    description = "Identificación automática de aves desde audio WAV/MP3.",
    version     = "1.0.0"
)

# CORS: permite llamadas desde cualquier dominio (incluyendo shinyapps.io)
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
    Verifica que la API está activa.
    Shiny llama esto al cargar la pestaña para despertar el servidor
    antes de que el usuario suba un audio (evita esperar 30s al analizar).
    """
    return {
        "status"       : "ok",
        "modelo_listo" : analyzer is not None,
        "mensaje"      : "API BirdNET activa y lista."
    }


# ── Análisis de audio ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_audio(
    audio    : UploadFile        = File(...),
    lat      : Optional[float]   = Form(None),
    lon      : Optional[float]   = Form(None),
    week     : Optional[int]     = Form(None),
    min_conf : float             = Form(0.10),
):
    """
    Analiza un archivo WAV o MP3 y devuelve las especies de aves detectadas.

    Parámetros opcionales:
      lat, lon  → coordenadas del lugar de grabación (mejoran la precisión)
      week      → semana del año 1-48 (filtra especies por época)
      min_conf  → confianza mínima 0-1 (default 0.10)

    Para grabaciones en Chile: lat entre -17 y -56, lon entre -66 y -75.
    """

    if analyzer is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Modelo no disponible. Intenta en unos segundos."
        )

    # Validar formato del archivo
    extension = os.path.splitext(audio.filename or "audio.wav")[-1].lower()
    if extension not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(
            status_code = 400,
            detail      = f"Formato '{extension}' no soportado. Usa WAV o MP3."
        )

    if not (0 <= min_conf <= 1):
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
                raise HTTPException(status_code=400, detail="Archivo vacío.")
            tmp.write(contenido)
            tmp_path = tmp.name

        # ── Análisis con birdnetlib ────────────────────────────────────────────
        # Recording acepta lat/lon/week opcionales para filtrar por rango
        # geográfico-temporal, reduciendo falsos positivos significativamente.
        recording = Recording(
            analyzer,
            tmp_path,
            lat      = lat,
            lon      = lon,
            week     = week if week else -1,
            min_conf = min_conf,
            overlap  = 0.0,
        )
        recording.analyze()

        detecciones_raw = recording.detections  # lista de dicts

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

        # Ordenar: primero por tiempo, luego por confianza descendente
        detecciones.sort(key=lambda x: (x["inicio_seg"], -x["confianza"]))

        # Resumen por especie única (tabla principal en Shiny)
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
            "especies"          : especies,      # una fila por especie
            "detecciones"       : detecciones,   # una fila por segmento de 3s
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
        # Borrar siempre el archivo temporal
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

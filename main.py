"""
BirdNET Detection API — backend para identificación automática de aves.
Usa birdnetlib + tflite-runtime (Python 3.10, ~10 MB, cabe en tier free).
Desplegado en Render.com.

Dependencias en requirements.txt:
  birdnetlib, librosa, tflite-runtime, fastapi, uvicorn, python-multipart
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import traceback
from typing import Optional


# ── Estado global del modelo ───────────────────────────────────────────────────
MODEL = {"analyzer": None, "listo": False, "error": None}


# ── Lifespan: carga el modelo después de que uvicorn abre el puerto ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("==> Iniciando servidor BirdNET API...")
    try:
        from birdnetlib.analyzer import Analyzer
        print("==> Cargando modelo BirdNET con tflite-runtime...")
        MODEL["analyzer"] = Analyzer()
        MODEL["listo"]    = True
        print("==> Modelo BirdNET cargado correctamente.")
    except Exception as e:
        MODEL["error"] = str(e)
        MODEL["listo"] = False
        print(f"==> ERROR al cargar modelo: {e}")
        traceback.print_exc()

    yield

    print("==> Apagando servidor BirdNET API.")


# ── Aplicación FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title    = "BirdNET Detection API",
    version  = "1.0.0",
    lifespan = lifespan,
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
    return {
        "status"       : "ok" if MODEL["listo"] else "iniciando",
        "modelo_listo" : MODEL["listo"],
        "error"        : MODEL["error"],
        "mensaje"      : (
            "API BirdNET activa y lista."
            if MODEL["listo"]
            else "Modelo cargando, intenta en 30 segundos."
        ),
    }


@app.get("/ping")
def ping():
    return {"pong": True, "modelo_listo": MODEL["listo"]}


# ── Análisis de audio ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_audio(
    audio    : UploadFile      = File(...),
    lat      : Optional[float] = Form(None),
    lon      : Optional[float] = Form(None),
    min_conf : float           = Form(0.10),
):
    """
    Analiza un archivo WAV o MP3 y devuelve las especies detectadas.
    lat/lon opcionales — mejoran precisión filtrando por rango geográfico.
    """
    if not MODEL["listo"]:
        raise HTTPException(
            status_code = 503,
            detail      = (
                f"Modelo no disponible: {MODEL['error'] or 'cargando'}. "
                "Espera 30 segundos y reintenta."
            )
        )

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
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
            contenido = await audio.read()
            if len(contenido) == 0:
                raise HTTPException(status_code=400, detail="Archivo de audio vacío.")
            tmp.write(contenido)
            tmp_path = tmp.name

        print(f"[INFO] Analizando '{audio.filename}' "
              f"(lat={lat}, lon={lon}, min_conf={min_conf})")

        from birdnetlib import Recording

        # lat/lon solo se pasan si tienen valor — evita errores internos con None
        kwargs = {"min_conf": min_conf}
        if lat is not None and lon is not None:
            kwargs["lat"] = lat
            kwargs["lon"] = lon

        recording = Recording(MODEL["analyzer"], tmp_path, **kwargs)
        recording.analyze()

        detecciones_raw = recording.detections
        print(f"[INFO] Detecciones brutas: {len(detecciones_raw)}")

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
        print(f"[INFO] Especies detectadas: {len(especies)}")

        return {
            "ok"            : True,
            "archivo"       : audio.filename,
            "n_detecciones" : len(detecciones),
            "n_especies"    : len(especies),
            "filtros"       : {"lat": lat, "lon": lon, "min_conf": min_conf},
            "especies"      : especies,
            "detecciones"   : detecciones,
        }

    except HTTPException:
        raise

    except Exception as e:
        msg = f"Error al procesar el audio: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=msg)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

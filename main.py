"""
BirdNET-Analyzer API — backend para detección automática de aves.
Desplegado en Render.com, consumido desde una app Shiny via httr.
 
Endpoints:
  GET  /          → health check (útil para el "ping" anti-sleep de Shiny)
  POST /analyze   → recibe audio, devuelve detecciones de especies
"""
 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import traceback
from typing import Optional
 
# ── BirdNET ────────────────────────────────────────────────────────────────────
# birdnet-analyzer expone una API Python limpia desde v2.x.
# El modelo TFLite (~300 MB) se descarga automáticamente en el primer arranque
# y queda cacheado en disco — no se vuelve a descargar en peticiones siguientes.
from birdnet import SpeciesPredictor, AudioAnalyzer
 
# ── Inicialización del modelo (una sola vez al arrancar el servidor) ────────────
# Esto es crítico: cargar el modelo aquí evita hacerlo en cada petición,
# lo que reduciría el tiempo de respuesta de ~30s a ~2-5s por análisis.
print("Cargando modelo BirdNET... (puede tardar en el primer arranque)")
try:
    analyzer = AudioAnalyzer()
    print("Modelo BirdNET cargado correctamente.")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    analyzer = None
 
# ── Aplicación FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(
    title       = "BirdNET Detection API",
    description = "Detección automática de especies de aves a partir de audio.",
    version     = "1.0.0"
)
 
# CORS: permite que Shiny (en cualquier dominio) pueda llamar a esta API.
# En producción podrías restringir origins a tu dominio de shinyapps.io.
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
    Endpoint de estado. Shiny puede llamar esto al cargar la pestaña
    para 'despertar' el servidor de Render antes de que el usuario analice.
    """
    return {
        "status"       : "ok",
        "modelo_listo" : analyzer is not None,
        "mensaje"       : "API BirdNET activa y lista para recibir audio."
    }
 
 
# ── Análisis de audio ──────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze_audio(
    audio    : UploadFile = File(..., description="Archivo WAV o MP3 a analizar."),
    lat      : Optional[float] = Form(None,  description="Latitud decimal (ej: -33.45). Mejora la precisión filtrando por rango geográfico."),
    lon      : Optional[float] = Form(None,  description="Longitud decimal (ej: -70.65)."),
    week     : Optional[int]   = Form(None,  description="Semana del año 1-48. Si se omite, BirdNET no filtra por época."),
    min_conf : float           = Form(0.10,  description="Confianza mínima 0-1. Valores bajos = más detecciones, más falsos positivos."),
):
    """
    Analiza un archivo de audio y devuelve las especies detectadas.
 
    Parámetros opcionales (lat, lon, week) activan el filtro de rango
    geográfico-temporal de BirdNET, reduciendo falsos positivos
    significativamente. Para Chile: lat ≈ -17 a -56, lon ≈ -66 a -75.
 
    Devuelve una lista de detecciones ordenadas por tiempo de inicio,
    cada una con: especie científica, nombre común, confianza, y el
    segmento temporal dentro del audio donde fue detectada.
    """
 
    # Validaciones básicas
    if analyzer is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Modelo BirdNET no disponible. Intenta nuevamente en unos segundos."
        )
 
    extension = os.path.splitext(audio.filename)[-1].lower()
    if extension not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(
            status_code = 400,
            detail      = f"Formato '{extension}' no soportado. Usa WAV, MP3, FLAC, OGG o M4A."
        )
 
    if min_conf < 0 or min_conf > 1:
        raise HTTPException(
            status_code = 400,
            detail      = "min_conf debe estar entre 0 y 1."
        )
 
    # Guardar el audio en un archivo temporal
    # tempfile.NamedTemporaryFile con delete=False porque BirdNET necesita
    # leer el archivo por ruta (no por stream).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete = False,
            suffix = extension
        ) as tmp:
            contenido = await audio.read()
            tmp.write(contenido)
            tmp_path = tmp.name
 
        # Verificar que el archivo no esté vacío
        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(status_code=400, detail="El archivo de audio está vacío.")
 
        # ── Análisis con BirdNET ───────────────────────────────────────────────
        # analyzer.predict() devuelve una lista de detecciones.
        # Cada detección es un dict con: start, end, common_name,
        # scientific_name, confidence.
        resultados_raw = analyzer.predict(
            audio_path = tmp_path,
            lat        = lat,
            lon        = lon,
            week       = week,
            min_conf   = min_conf,
        )
 
        # ── Formatear respuesta ────────────────────────────────────────────────
        # Filtrar por confianza mínima y estructurar para JSON limpio.
        detecciones = []
        for det in resultados_raw:
            confianza = round(float(det.get("confidence", 0)), 4)
            if confianza < min_conf:
                continue
 
            detecciones.append({
                "inicio_seg"     : round(float(det.get("start_time", 0)), 2),
                "fin_seg"        : round(float(det.get("end_time",   3)), 2),
                "nombre_cientifico" : det.get("scientific_name", ""),
                "nombre_comun"   : det.get("common_name",     ""),
                "confianza"      : confianza,
                "confianza_pct"  : f"{confianza * 100:.1f}%",
            })
 
        # Ordenar por tiempo de inicio, luego por confianza descendente
        detecciones.sort(key=lambda x: (x["inicio_seg"], -x["confianza"]))
 
        # Resumen de especies únicas detectadas (para mostrar en la UI)
        especies_unicas = list({
            d["nombre_cientifico"]: {
                "nombre_cientifico" : d["nombre_cientifico"],
                "nombre_comun"      : d["nombre_comun"],
                "max_confianza"     : max(
                    x["confianza"] for x in detecciones
                    if x["nombre_cientifico"] == d["nombre_cientifico"]
                ),
                "n_detecciones"     : sum(
                    1 for x in detecciones
                    if x["nombre_cientifico"] == d["nombre_cientifico"]
                ),
            }
            for d in detecciones
        }.values())
 
        especies_unicas.sort(key=lambda x: -x["max_confianza"])
 
        return {
            "ok"               : True,
            "archivo"          : audio.filename,
            "n_detecciones"    : len(detecciones),
            "n_especies"       : len(especies_unicas),
            "filtros_aplicados": {
                "lat"     : lat,
                "lon"     : lon,
                "week"    : week,
                "min_conf": min_conf,
            },
            "especies"         : especies_unicas,   # resumen por especie
            "detecciones"      : detecciones,        # todas las detecciones con tiempo
        }
 
    except HTTPException:
        # Re-lanzar excepciones HTTP tal cual (son errores controlados)
        raise
 
    except Exception as e:
        # Error inesperado — loguear en consola de Render y devolver 500
        print(f"[ERROR] Análisis fallido para '{audio.filename}':")
        traceback.print_exc()
        raise HTTPException(
            status_code = 500,
            detail      = f"Error al analizar el audio: {str(e)}"
        )
 
    finally:
        # Siempre borrar el archivo temporal, incluso si hubo error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

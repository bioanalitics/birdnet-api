"""
BirdNET Detection API — Python 3.10, tflite-runtime, numpy<2.0
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, traceback
from typing import Optional

MODEL = {"analyzer": None, "listo": False, "error": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("==> Cargando modelo BirdNET...")
    try:
        from birdnetlib.analyzer import Analyzer
        MODEL["analyzer"] = Analyzer()
        MODEL["listo"] = True
        print("==> Modelo cargado.")
    except Exception as e:
        MODEL["error"] = str(e)
        print(f"==> ERROR: {e}")
        traceback.print_exc()
    yield

app = FastAPI(title="BirdNET API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health_check():
    return {"status": "ok" if MODEL["listo"] else "iniciando",
            "modelo_listo": MODEL["listo"], "error": MODEL["error"]}

@app.get("/ping")
def ping():
    return {"pong": True, "modelo_listo": MODEL["listo"]}

@app.post("/analyze")
async def analyze_audio(
    audio    : UploadFile      = File(...),
    lat      : Optional[float] = Form(None),
    lon      : Optional[float] = Form(None),
    min_conf : float           = Form(0.10),
):
    if not MODEL["listo"]:
        raise HTTPException(503, detail=f"Modelo no listo: {MODEL['error'] or 'cargando'}")

    ext = os.path.splitext(audio.filename or "audio.wav")[-1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        raise HTTPException(400, detail=f"Formato '{ext}' no soportado.")

    tmp_path = tmp_wav = None
    try:
        # Guardar audio original
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            data = await audio.read()
            if not data:
                raise HTTPException(400, detail="Archivo vacío.")
            tmp.write(data)
            tmp_path = tmp.name

        # ── Recortar a máximo 60 segundos con pydub (ya instalado via birdnetlib) ──
        # Esto evita timeouts en audios largos en el tier free de Render.
        try:
            from pydub import AudioSegment
            MAX_MS = 60_000  # 60 segundos
            seg = AudioSegment.from_file(tmp_path)
            if len(seg) > MAX_MS:
                print(f"[INFO] Audio de {len(seg)/1000:.1f}s recortado a 60s")
                seg = seg[:MAX_MS]
            tmp_wav = tempfile.mktemp(suffix=".wav")
            seg.export(tmp_wav, format="wav")
            analysis_path = tmp_wav
        except Exception as e:
            print(f"[WARN] pydub falló ({e}), usando archivo original")
            analysis_path = tmp_path

        print(f"[INFO] Analizando '{audio.filename}' lat={lat} lon={lon} min_conf={min_conf}")

        from birdnetlib import Recording
        kwargs = {"min_conf": min_conf}
        if lat is not None and lon is not None:
            kwargs["lat"] = lat
            kwargs["lon"] = lon

        recording = Recording(MODEL["analyzer"], analysis_path, **kwargs)
        recording.analyze()

        raw = recording.detections
        print(f"[INFO] Detecciones: {len(raw)}")

        detecciones = sorted([{
            "inicio_seg"        : round(float(d.get("start_time", 0)), 2),
            "fin_seg"           : round(float(d.get("end_time",   3)), 2),
            "nombre_cientifico" : d.get("scientific_name", ""),
            "nombre_comun"      : d.get("common_name",     ""),
            "confianza"         : round(float(d.get("confidence", 0)), 4),
            "confianza_pct"     : f"{float(d.get('confidence',0))*100:.1f}%",
        } for d in raw], key=lambda x: (x["inicio_seg"], -x["confianza"]))

        vistas = {}
        for d in detecciones:
            sp = d["nombre_cientifico"]
            if sp not in vistas:
                vistas[sp] = {"nombre_cientifico": sp, "nombre_comun": d["nombre_comun"],
                              "max_confianza": d["confianza"],
                              "max_confianza_pct": d["confianza_pct"], "n_detecciones": 1}
            else:
                vistas[sp]["n_detecciones"] += 1
                if d["confianza"] > vistas[sp]["max_confianza"]:
                    vistas[sp]["max_confianza"] = d["confianza"]
                    vistas[sp]["max_confianza_pct"] = d["confianza_pct"]

        especies = sorted(vistas.values(), key=lambda x: -x["max_confianza"])
        print(f"[INFO] Especies: {len(especies)}")

        return {"ok": True, "archivo": audio.filename,
                "n_detecciones": len(detecciones), "n_especies": len(especies),
                "filtros": {"lat": lat, "lon": lon, "min_conf": min_conf},
                "especies": especies, "detecciones": detecciones}

    except HTTPException:
        raise
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] {msg}")
        traceback.print_exc()
        raise HTTPException(500, detail=f"Error al procesar audio: {msg}")
    finally:
        for p in [tmp_path, tmp_wav]:
            if p and os.path.exists(p):
                os.unlink(p)

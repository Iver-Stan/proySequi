"""API REST (FastAPI).  Ejecutar:  uvicorn app.main:app --reload   ->  http://127.0.0.1:8000/docs"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException, Query  # noqa: E402

from src import predecir as PR  # noqa: E402

app = FastAPI(title="Riesgo de sequía en el Altiplano boliviano",
              description=PR.AVISO, version="1.0")


@lru_cache(maxsize=1)
def _art():
    return PR.cargar_artefacto()


@app.get("/health")
def health():
    try:
        a = _art()
        return {"estado": "ok", "version_modelo": a.get("version"), "modelo": a.get("modelo"),
                "modelos_por_horizonte": {h: v.get("nombre", a.get("modelo")) for h, v in a["modelos"].items()},
                "sintetico": a.get("sintetico", False)}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.get("/puntos")
def puntos():
    return {n: {"lat": la, "lon": lo} for n, (la, lo) in _art()["puntos"].items()}


@lru_cache(maxsize=64)
def _predecir_cache(punto, lat, lon, horizonte, dia):
    # 'dia' en la clave: la predicción se recalcula como máximo una vez al día
    return PR.predecir_punto(punto=punto, lat=lat, lon=lon, horizonte=horizonte, art=_art())


@app.get("/predict")
def predict(punto: str | None = Query(None, max_length=40, description="Nombre de un punto entrenado (ver /puntos)"),
            lat: float | None = Query(None, ge=-90.0, le=90.0, description="Latitud (-90 a 90)"),
            lon: float | None = Query(None, ge=-180.0, le=180.0, description="Longitud (-180 a 180)"),
            horizonte: int = Query(1, ge=1, le=3, description="Meses hacia adelante (1-3)")):
    if punto is None and (lat is None or lon is None):
        raise HTTPException(422, "Indica 'punto' o bien 'lat' y 'lon'.")
    lat_r = round(lat, 2) if lat is not None else None
    lon_r = round(lon, 2) if lon is not None else None
    try:
        import datetime as dt
        return _predecir_cache(punto, lat_r, lon_r, horizonte, dt.date.today().isoformat())
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(502, "No se pudo completar la consulta con el servicio satelital externo.")

"""Predicción con el modelo ya entrenado (usado por la API y por la app Streamlit).

Flujo: descarga los datos recientes de NASA POWER -> limpia -> mensualiza -> SPI con los
parámetros guardados del periodo de referencia -> variables -> probabilidad de sequía.
"""
import warnings
import math
import pandas as pd
import joblib

from . import config as C
from . import datos, spi as spi_mod, features as F

RUTA_ARTEFACTO = C.MODELOS / "artefacto_sequia.joblib"
AVISO = ("Herramienta académica y experimental basada en datos satelitales/reanálisis (NASA POWER). "
         "NO es un pronóstico oficial de SENAMHI.")


def guardar_artefacto(art, ruta=RUTA_ARTEFACTO):
    joblib.dump(art, ruta, compress=3)
    return ruta


def cargar_artefacto(ruta=RUTA_ARTEFACTO):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta primero el notebook (sección final) "
                                f"o 'python scripts/entrenar_y_guardar.py'.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(ruta)


def _riesgo(p, umbral):
    if p >= umbral:
        return "ALTO"
    if p >= umbral / 2:
        return "MEDIO"
    return "BAJO"


def _mensual_completo(lat, lon, nombre, ini, fin):
    d = datos.descargar_power_punto(nombre, lat, lon, ini=ini, fin=fin)
    dia, _ = datos.limpiar_diario(d)
    ultimo_dia = dia.index.max()
    m = datos.a_mensual(dia)
    fin_mes = m["fecha"] + pd.offsets.MonthEnd(0)
    return m[fin_mes <= ultimo_dia].reset_index(drop=True)      # solo meses COMPLETOS


def predecir_punto(punto=None, lat=None, lon=None, horizonte=1, art=None, hoy=None):
    """Predice la probabilidad de sequía (SPI-3 <= -1) 'horizonte' meses después del último mes completo.
    - punto: nombre de uno de los puntos entrenados, o bien lat/lon para un punto nuevo
      (en ese caso se descarga 1981-hoy para ajustar el SPI de ese punto: tarda más y es MENOS fiable)."""
    art = art or cargar_artefacto()
    hoy = pd.Timestamp(hoy) if hoy is not None else pd.Timestamp.today().normalize()
    if horizonte not in art["modelos"]:
        raise ValueError(f"Horizonte no disponible: {horizonte}. Opciones: {sorted(art['modelos'])}")

    nuevo = punto is None or punto not in art["puntos"]
    if not nuevo:
        lat, lon = art["puntos"][punto]
        ini = (hoy - pd.DateOffset(months=26)).replace(day=1).strftime("%Y%m%d")
        m = _mensual_completo(lat, lon, punto, ini, hoy.strftime("%Y%m%d"))
        nombre = punto
        params, clim = art["spi_params"], art["climatologia"]
    else:
        if lat is None or lon is None:
            raise ValueError("Indica un punto entrenado o lat y lon.")
        lat, lon = float(lat), float(lon)
        if not (math.isfinite(lat) and math.isfinite(lon) and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("Coordenadas fuera de rango geográfico válido (-90<=lat<=90, -180<=lon<=180).")
        nombre = f"punto_nuevo({lat:.2f},{lon:.2f})"
        m = _mensual_completo(lat, lon, nombre, C.FECHA_INI, hoy.strftime("%Y%m%d"))
        params = {k: spi_mod.ajustar_parametros(m, k) for k in C.ESCALAS_SPI}
        clim = F.calcular_climatologia(m)

    ms = spi_mod.agregar_spi(m, params)
    oni = datos.descargar_oni()
    d = F.construir_features(ms, oni, clim, horizontes=[horizonte])

    cfg = art["modelos"][horizonte]
    feats = cfg["features"]
    trat = cfg.get("tratamiento", "ninguno")
    if trat not in ("ninguno", "log1p"):
        raise ValueError("El despliegue solo soporta tratamiento 'ninguno' o 'log1p'.")
    d = F.tratar_outliers(d, trat)
    listo = d.dropna(subset=feats)
    if listo.empty:
        raise RuntimeError("No hay un mes reciente con todas las variables disponibles "
                           "(¿falta el ONI reciente o datos de POWER?).")
    fila = listo.iloc[[-1]]
    p = float(cfg["pipeline"].predict_proba(fila[feats])[:, 1][0])
    t = pd.Timestamp(fila["fecha"].iloc[0])
    nombre_modelo = cfg.get("nombre", art.get("modelo"))
    nota_climatologia = (" Horizonte 3: referencia climatológica; no supera de forma consistente "
                         "a la persistencia." if nombre_modelo == "Climatología" else "")
    return {
        "punto": nombre, "lat": lat, "lon": lon, "horizonte_meses": horizonte,
        "mes_base": t.strftime("%Y-%m"),
        "mes_objetivo": (t + pd.DateOffset(months=horizonte)).strftime("%Y-%m"),
        "probabilidad_sequia": round(p, 4),
        "umbral_modelo": round(cfg["umbral"], 4),
        "riesgo": _riesgo(p, cfg["umbral"]),
        "spi3_actual": None if pd.isna(fila["spi3"].iloc[0]) else round(float(fila["spi3"].iloc[0]), 2),
        "lluvia_mes_base_mm": round(float(fila["lluvia"].iloc[0]), 1),
        "punto_nuevo": bool(nuevo),
        "version_modelo": art.get("version"),
        "modelo_horizonte": nombre_modelo,
        "aviso": AVISO + nota_climatologia +
                 (" Punto fuera del conjunto de entrenamiento: fiabilidad menor." if nuevo else ""),
    }


def predecir_todos(horizonte=1, art=None, hoy=None):
    art = art or cargar_artefacto()
    filas = []
    for p in art["puntos"]:
        try:
            filas.append(predecir_punto(p, horizonte=horizonte, art=art, hoy=hoy))
        except Exception as e:
            filas.append({"punto": p, "error": str(e)})
    return filas

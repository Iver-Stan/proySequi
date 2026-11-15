"""Descarga (NASA POWER, NOAA ONI), limpieza y agregación mensual."""
import re
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

from . import config as C

# =============================================================== NASA POWER
def descargar_power_punto(nombre, lat, lon, ini=None, fin=None, reintentos=3):
    """Descarga datos DIARIOS de un punto. Devuelve DataFrame indexado por fecha
    con las columnas originales de POWER (con -999 como relleno) + 'altitud' + 'punto'."""
    ini, fin = ini or C.FECHA_INI, fin or C.FECHA_FIN
    if C.MODO_SINTETICO:
        from .sintetico import diario_sintetico
        d = diario_sintetico(nombre, lat, lon, pd.to_datetime(ini), pd.to_datetime(fin))
        d["punto"] = nombre
        return d

    params = {"parameters": ",".join(C.PARAMS_POWER), "community": "AG",
              "latitude": lat, "longitude": lon, "start": ini, "end": fin, "format": "JSON"}
    ultimo_error = None
    for intento in range(1, reintentos + 1):
        try:
            r = requests.get(C.POWER_URL, params=params, timeout=180)
            r.raise_for_status()
            j = r.json()
            d = pd.DataFrame(j["properties"]["parameter"])
            d.index = pd.to_datetime(d.index, format="%Y%m%d")
            d.index.name = "fecha"
            d["altitud"] = float(j["geometry"]["coordinates"][2])
            d["punto"] = nombre
            return d
        except Exception as e:                      # red, JSON, límite de uso...
            ultimo_error = e
            time.sleep(5 * intento)
    raise RuntimeError(f"No se pudo descargar '{nombre}' de NASA POWER tras {reintentos} intentos: {ultimo_error}")


def descargar_todos(puntos=None, forzar=False, archivo="power_diario.csv"):
    """Descarga todos los puntos y guarda un CSV crudo (caché). Si ya existe, lo reutiliza."""
    ruta = C.DATA_RAW / archivo
    if ruta.exists() and not forzar:
        return pd.read_csv(ruta, parse_dates=["fecha"], index_col="fecha")
    puntos = puntos or C.PUNTOS
    partes = []
    for nombre, (lat, lon) in puntos.items():
        print(f"  descargando {nombre} ({lat}, {lon}) ...")
        partes.append(descargar_power_punto(nombre, lat, lon))
    df = pd.concat(partes)
    df.to_csv(ruta)
    return df


# =============================================================== ONI
MESES_TRIM = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _parsear_oni_html(html):
    """La página ONI v6 de NOAA es una tabla HTML: Año + 12 trimestres móviles."""
    texto = re.sub(r"<[^>]+>", " ", html)
    texto = texto.replace("**", " ")
    filas = {}
    for m in re.finditer(r"\b(19[5-9]\d|20[0-4]\d)\b((?:\s+-?\d{1,2}\.\d){1,12})", texto):
        anio = int(m.group(1))
        vals = [float(v) for v in m.group(2).split()]
        if anio not in filas or len(vals) > len(filas[anio]):
            filas[anio] = vals
    return _filas_a_df(filas)


def _parsear_oni_txt(txt):
    """Formato clásico oni.ascii.txt:  SEAS  YR  TOTAL  ANOM"""
    filas = {}
    for linea in txt.splitlines():
        p = linea.split()
        if len(p) >= 4 and p[0] in MESES_TRIM:
            anio, anom = int(p[1]), float(p[3])
            filas.setdefault(anio, [np.nan] * 12)[MESES_TRIM.index(p[0])] = anom
    return _filas_a_df(filas)


def _filas_a_df(filas):
    reg = []
    for anio, vals in filas.items():
        for i, v in enumerate(vals):
            # el trimestre i está CENTRADO en el mes i+1 (DJF -> enero)
            reg.append({"fecha": pd.Timestamp(year=anio, month=i + 1, day=1), "oni": v})
    df = pd.DataFrame(reg).dropna().sort_values("fecha").reset_index(drop=True)
    if len(df) < 500:
        raise ValueError(f"Tabla ONI sospechosamente corta ({len(df)} filas).")
    return df


def descargar_oni(forzar=False):
    """Devuelve DataFrame [fecha (mes central), oni]. Intenta: caché -> HTML v6 -> TXT -> archivo manual."""
    ruta = C.DATA_RAW / "oni.csv"
    if C.MODO_SINTETICO:
        from .sintetico import oni_sintetico
        return oni_sintetico()
    if ruta.exists() and not forzar:
        return pd.read_csv(ruta, parse_dates=["fecha"])

    errores = []
    intentos = [("html", C.ONI_URL_HTML, _parsear_oni_html)] + \
               [("txt", u, _parsear_oni_txt) for u in C.ONI_URLS_TXT]
    for tipo, url, parser in intentos:
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "proyecto-sequia-bolivia/1.0"})
            r.raise_for_status()
            df = parser(r.text)
            df.to_csv(ruta, index=False)
            return df
        except Exception as e:
            errores.append(f"{url}: {e}")
    if C.ONI_ARCHIVO_MANUAL.exists():                 # plan B manual
        txt = C.ONI_ARCHIVO_MANUAL.read_text(encoding="utf-8", errors="ignore")
        es_txt = re.search(r"^\s*(?:DJF|JFM|FMA|MAM|AMJ|MJJ|JJA|JAS|ASO|SON|OND|NDJ)\s+\d{4}\s", txt, re.M) is not None
        df = _parsear_oni_txt(txt) if es_txt else _parsear_oni_html(txt)
        df.to_csv(ruta, index=False)
        return df
    raise RuntimeError("No se pudo obtener el ONI.\n" + "\n".join(errores) +
                       f"\nPlan B: abre {C.ONI_URL_HTML}, guarda la página como "
                       f"'{C.ONI_ARCHIVO_MANUAL}' y vuelve a ejecutar.")


# =============================================================== LIMPIEZA
RANGOS_FISICOS = {            # (mínimo, máximo) físicamente posibles para estas variables
    "lluvia": (0, 500), "t2m": (-40, 45), "tmax": (-30, 55), "tmin": (-50, 40),
    # ALLSKY_SFC_SW_DWN de POWER (community=AG) viene en MJ/m²/día, no en kWh/m²/día.
    "rh2m": (0, 100), "ws2m": (0, 60), "ps": (40, 110), "rad": (0, 50),
}
VARS = list(RANGOS_FISICOS)


def limpiar_diario(df):
    """-999 -> NaN, renombra columnas, anula valores físicamente imposibles.
    Devuelve (df_limpio, reporte)."""
    d = df.rename(columns=C.RENOMBRAR).copy()
    rep = {"filas": len(d)}
    n999 = int((d[VARS] == -999).sum().sum())
    d[VARS] = d[VARS].replace(-999, np.nan)
    rep["valores_-999"] = n999
    imposibles = {}
    for v, (lo, hi) in RANGOS_FISICOS.items():
        m = (d[v] < lo) | (d[v] > hi)
        imposibles[v] = int(m.sum())
        d.loc[m, v] = np.nan
    inc = d["tmin"] > d["tmax"]
    imposibles["tmin>tmax"] = int(inc.sum())
    d.loc[inc, ["tmin", "tmax"]] = np.nan
    rep["imposibles_a_NaN"] = imposibles
    rep["duplicados_fecha_punto"] = int(d.reset_index().duplicated(["fecha", "punto"]).sum())
    return d, rep


def a_mensual(diario, cobertura_min=0.9):
    """Agrega diario -> mensual por punto.
    lluvia: suma (se reescala si faltan pocos días); resto: promedio; tmin: mínimo; tmax: máximo.
    Un mes con cobertura < cobertura_min queda NaN (incluye el mes en curso, incompleto)."""
    d = diario.copy()
    d["mes"] = d.index.to_period("M")
    filas = []
    for punto, g in d.groupby("punto"):
        gm = g.groupby("mes")
        dias = gm.size()
        out = pd.DataFrame({
            "lluvia": gm["lluvia"].mean() * dias.index.days_in_month,
            "t2m": gm["t2m"].mean(), "tmax": gm["tmax"].max(), "tmin": gm["tmin"].min(),
            "rh2m": gm["rh2m"].mean(), "ws2m": gm["ws2m"].mean(),
            "ps": gm["ps"].mean(), "rad": gm["rad"].mean(),
        })
        cobertura = gm["lluvia"].count() / dias.index.days_in_month
        out.loc[cobertura < cobertura_min, :] = np.nan
        out["cobertura_lluvia"] = cobertura
        out["altitud"] = g["altitud"].iloc[0]
        out["punto"] = punto
        out["fecha"] = out.index.to_timestamp()
        filas.append(out.reset_index(drop=True))
    m = pd.concat(filas, ignore_index=True)
    # malla mensual completa por punto (para que rezagos y rolling no salten meses)
    mallas = []
    for punto, g in m.groupby("punto"):
        idx = pd.date_range(g["fecha"].min(), g["fecha"].max(), freq="MS")
        g = g.set_index("fecha").reindex(idx)
        g["punto"] = punto
        g["altitud"] = g["altitud"].ffill().bfill()
        g.index.name = "fecha"
        mallas.append(g.reset_index())
    return pd.concat(mallas, ignore_index=True).sort_values(["punto", "fecha"]).reset_index(drop=True)

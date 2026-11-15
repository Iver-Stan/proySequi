"""
Generador de datos SINTÉTICOS.

!!! SOLO sirve para comprobar que el código corre sin internet.
!!! NO representa el clima real de Bolivia. No uses sus resultados en el informe.

Se activa con la variable de entorno SEQUIA_SINTETICO=1.
"""
import zlib

import numpy as np
import pandas as pd


def _oscilacion_enso(fechas):
    """Señal lenta tipo ENSO (periodo ~4 años) usada para el ONI y para modular la lluvia."""
    t = (fechas - pd.Timestamp("1981-01-01")).days.values / 365.25
    return 1.2 * np.sin(2 * np.pi * t / 4.0) + 0.5 * np.sin(2 * np.pi * t / 7.3 + 1.0)


def oni_sintetico(ini="1981-01-01", fin="2026-12-31"):
    f = pd.date_range(ini, fin, freq="MS")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"fecha": f, "oni": _oscilacion_enso(f) + rng.normal(0, 0.15, len(f))})


def diario_sintetico(nombre, lat, lon, ini, fin, semilla=0):
    rng = np.random.default_rng(zlib.crc32(nombre.encode()) if semilla == 0 else semilla)
    f = pd.date_range(ini, fin, freq="D")
    n = len(f)
    doy = f.day_of_year.values
    altitud = 3600 + (-lat - 16) * 80 + rng.normal(0, 40)

    # lluvia: temporada lluviosa dic-mar, modulada por "ENSO" (Niño -> menos lluvia en el Altiplano)
    estacional = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365.25)      # pico a mitad de enero
    enso = _oscilacion_enso(f)
    factor = np.clip(1 - 0.30 * enso, 0.2, 2.0)
    p_lluvia = np.clip(0.60 * estacional ** 1.5 * factor, 0, 0.9) + 0.01
    llueve = rng.random(n) < p_lluvia
    monto = rng.gamma(shape=0.9, scale=5.5 * factor, size=n)
    lluvia = np.where(llueve, monto, 0.0)
    # algunos extremos reales
    idx_ext = rng.choice(n, size=max(3, n // 4000), replace=False)
    lluvia[idx_ext] += rng.gamma(4, 15, size=len(idx_ext))

    t2m = 8.5 + 3.0 * np.sin(2 * np.pi * (doy - 100) / 365.25) + 0.4 * enso + rng.normal(0, 1.2, n) - (altitud - 3700) / 160
    tmax = t2m + 7 + rng.normal(0, 1.0, n)
    tmin = t2m - 8 + rng.normal(0, 1.0, n) - 3 * (1 - estacional)
    rh = np.clip(35 + 30 * estacional + 6 * (lluvia > 0) + rng.normal(0, 6, n), 5, 100)
    ws = np.clip(3.2 + rng.normal(0, 0.8, n), 0.2, None)
    ps = 63 + rng.normal(0, 0.3, n)
    # POWER entrega la radiación diaria en MJ/m²/día (1 kWh = 3.6 MJ).
    rad = 3.6 * np.clip(6.5 + 1.0 * np.cos(2 * np.pi * (doy - 355) / 365.25) - 1.2 * (lluvia > 0) + rng.normal(0, 0.5, n), 0.5, None)

    d = pd.DataFrame({"PRECTOTCORR": lluvia, "T2M": t2m, "T2M_MAX": tmax, "T2M_MIN": tmin,
                      "RH2M": rh, "WS2M": ws, "PS": ps, "ALLSKY_SFC_SW_DWN": rad}, index=f)

    # suciedad realista para probar la limpieza: -999 y valores imposibles
    for c in d.columns:
        m = rng.random(n) < 0.004
        d.loc[m, c] = -999.0
    d.loc[d.index[rng.choice(n, 4, replace=False)], "RH2M"] = 130.0
    d.loc[d.index[rng.choice(n, 3, replace=False)], "PRECTOTCORR"] = -5.0
    d.index.name = "fecha"
    d["altitud"] = float(altitud)
    return d

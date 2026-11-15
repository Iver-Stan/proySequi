"""
Índice Estandarizado de Precipitación (SPI) con distribución gamma mixta.

Para cada punto y cada mes calendario se ajusta, sobre el periodo de referencia (1981-2010):
    q      = probabilidad de acumulado == 0
    a, esc = parámetros de la gamma ajustada a los acumulados > 0
    H(x)   = q + (1 - q) * Gamma_cdf(x)        SPI = Phi^-1( H(x) )

Los parámetros se guardan para poder calcular el SPI de datos NUEVOS en el despliegue.
Nota: en meses/puntos muy secos q es grande y un acumulado 0 ya produce SPI <= -1;
por eso se ofrece la alternativa de evaluar solo la temporada lluviosa (ver notebook).
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def acumulado(mensual, k, col="lluvia"):
    """Lluvia acumulada de k meses (ventana móvil que termina en el mes t) por punto."""
    return (mensual.sort_values(["punto", "fecha"])
            .groupby("punto")[col]
            .transform(lambda s: s.rolling(k, min_periods=k).sum()))


def _ajustar(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 15:
        return (np.nan, np.nan, np.nan)
    q = float((x <= 0).mean())
    pos = x[x > 0]
    if len(pos) < 8:
        return (q, np.nan, np.nan)
    a, _, esc = stats.gamma.fit(pos, floc=0)
    return (q, float(a), float(esc))


def ajustar_parametros(mensual, k, ref_ini=None, ref_fin=None):
    """Devuelve DataFrame con índice (punto, mes) y columnas q, a, esc."""
    ref_ini, ref_fin = ref_ini or C.REF_INI, ref_fin or C.REF_FIN
    d = mensual[["punto", "fecha"]].copy()
    d["acum"] = acumulado(mensual, k)
    d["mes"] = d["fecha"].dt.month
    d = d[(d["fecha"].dt.year >= ref_ini) & (d["fecha"].dt.year <= ref_fin)]
    filas = []
    for (punto, mes), g in d.groupby(["punto", "mes"]):
        q, a, esc = _ajustar(g["acum"].values)
        filas.append({"punto": punto, "mes": mes, "q": q, "a": a, "esc": esc})
    return pd.DataFrame(filas).set_index(["punto", "mes"])


def calcular_spi(mensual, k, params):
    """SPI-k para cada fila de 'mensual' usando 'params' (de ajustar_parametros)."""
    d = mensual[["punto", "fecha"]].copy()
    d["x"] = acumulado(mensual, k)
    d["mes"] = d["fecha"].dt.month
    d = d.join(params, on=["punto", "mes"])
    cdf_g = stats.gamma.cdf(d["x"].values, a=d["a"].values, scale=d["esc"].values)
    H = d["q"].values + (1 - d["q"].values) * cdf_g
    # acumulado 0 -> H = q ; si la gamma no pudo ajustarse (a NaN) el SPI queda NaN
    H = np.where(d["x"].values <= 0, d["q"].values, H)
    H = np.clip(H, 1e-4, 1 - 1e-4)
    spi = stats.norm.ppf(H)
    x = d["x"].values
    invalido = np.isnan(x) | np.isnan(d["q"].values) | ((x > 0) & np.isnan(d["a"].values))
    spi = np.where(invalido, np.nan, spi)
    return pd.Series(np.clip(spi, -3, 3), index=mensual.index, name=f"spi{k}")


def agregar_spi(mensual, params_por_escala):
    """Añade columnas spi1, spi3, spi6 al DataFrame mensual."""
    out = mensual.copy()
    for k, p in params_por_escala.items():
        out[f"spi{k}"] = calcular_spi(out, k, p)
    return out

"""Ingeniería de variables, etiquetas, particiones temporales y tratamiento de outliers."""
import numpy as np
import pandas as pd

from . import config as C
from .spi import acumulado

COLS_CLIMA = ["t2m", "tmin", "tmax", "rh2m"]
COLS_LLUVIA = ["lluvia", "lluvia_l1", "lluvia_l2", "lluvia_l3", "lluvia_l6", "acum3", "acum6"]

FEATURES_BASE = (
    COLS_LLUVIA
    + ["spi1", "spi3", "spi6", "spi1_l1", "spi3_l1", "spi3_l2", "spi3_l3"]
    + ["t2m", "tmax", "tmin", "rh2m", "ws2m", "ps", "rad"]
    + ["t2m_anom", "tmin_anom", "tmax_anom", "rh2m_anom"]
    + ["mes_sin", "mes_cos"]
)
FEATURES_ONI = [f"oni_l{l}" for l in C.ONI_REZAGOS]


def lista_features(con_oni=True, con_estaticas=True):
    f = list(FEATURES_BASE)
    if con_oni:
        f += FEATURES_ONI
    if con_estaticas:
        f += list(C.FEATURES_ESTATICAS)
    return f


# ------------------------------------------------------------------ climatología
def calcular_climatologia(mensual):
    """Promedio mensual de t2m/tmin/tmax/rh2m por (punto, mes) en el periodo de referencia."""
    d = mensual.copy()
    d["mes"] = d["fecha"].dt.month
    d = d[(d["fecha"].dt.year >= C.REF_INI) & (d["fecha"].dt.year <= C.REF_FIN)]
    clim = d.groupby(["punto", "mes"])[COLS_CLIMA].mean()
    clim.columns = [f"clim_{c}" for c in clim.columns]
    return clim


# ------------------------------------------------------------------ features
def construir_features(mensual_spi, oni, clim, horizontes=None):
    """Tabla (punto, fecha) con predictores en el mes t y etiquetas y_h{h} para t+h."""
    horizontes = horizontes or C.HORIZONTES
    d = mensual_spi.sort_values(["punto", "fecha"]).reset_index(drop=True).copy()
    d["mes"] = d["fecha"].dt.month

    def lag(col, n):
        return d.groupby("punto")[col].shift(n)

    for n in (1, 2, 3, 6):
        d[f"lluvia_l{n}"] = lag("lluvia", n)
    d["acum3"] = acumulado(d, 3)
    d["acum6"] = acumulado(d, 6)
    d["spi1_l1"] = lag("spi1", 1)
    for n in (1, 2, 3):
        d[f"spi3_l{n}"] = lag("spi3", n)

    d = d.join(clim, on=["punto", "mes"])
    for c in COLS_CLIMA:
        d[f"{c}_anom"] = d[c] - d[f"clim_{c}"]
    d = d.drop(columns=[f"clim_{c}" for c in COLS_CLIMA])

    # ONI: se publica con retraso (y se revisa hasta ~2 meses), por eso solo se usan rezagos >= 2
    oni_map = oni.set_index("fecha")["oni"]
    for l in C.ONI_REZAGOS:
        d[f"oni_l{l}"] = (d["fecha"] - pd.DateOffset(months=l)).map(oni_map)
    d["oni_t"] = d["fecha"].map(oni_map)          # SOLO para análisis (no es predictor)

    d["mes_sin"] = np.sin(2 * np.pi * d["mes"] / 12)
    d["mes_cos"] = np.cos(2 * np.pi * d["mes"] / 12)

    d["sequia_actual"] = np.where(d["spi3"].isna(), np.nan, (d["spi3"] <= C.UMBRAL_SEQUIA).astype(float))
    for h in horizontes:
        fut = d.groupby("punto")["spi3"].shift(-h)
        d[f"spi3_fut_h{h}"] = fut
        d[f"y_h{h}"] = np.where(fut.isna(), np.nan, (fut <= C.UMBRAL_SEQUIA).astype(float))
    return d


# ------------------------------------------------------------------ particiones
def particionar(d, h, features, con_gap=True):
    """Divide en train / val / test por FECHA (nunca al azar).
    La etiqueta de una fila t se conoce en t+h, por eso train y val se recortan para que
    su fecha OBJETIVO no invada el periodo siguiente (sin fuga temporal)."""
    x = d.dropna(subset=features + [f"y_h{h}"]).copy()
    per = x["fecha"].dt.to_period("M")
    obj = per + h
    tr_fin = pd.Period(C.TRAIN_FIN, "M")
    va_fin = pd.Period(C.VAL_FIN, "M")
    gap = obj if con_gap else per
    m_train = gap <= tr_fin
    m_val = (per > tr_fin) & (gap <= va_fin)
    m_test = per > va_fin
    return x[m_train], x[m_val], x[m_test]


def splits_panel(fechas, n_splits=5, gap_meses=0):
    """Validación cruzada temporal para datos de panel (varios puntos por fecha):
    divide las FECHAS únicas en orden y devuelve índices posicionales de las filas."""
    from sklearn.model_selection import TimeSeriesSplit
    fechas = pd.Series(pd.to_datetime(np.asarray(fechas)))
    unicas = np.sort(fechas.unique())
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap_meses)
    out = []
    for tr_u, te_u in tss.split(unicas):
        tr = np.where(fechas.isin(unicas[tr_u]))[0]
        te = np.where(fechas.isin(unicas[te_u]))[0]
        out.append((tr, te))
    return out


def splits_post_referencia(fechas, horizonte, inicio=2011, fin=2018, bloque_anios=2):
    """Pliegues temporales cuyo primer periodo de evaluación empieza después de 1981-2010.

    Los predictores SPI/climatología usan la referencia fija hasta 2010. Por eso
    no se evalúan pliegues anteriores a 2011: allí incluirían información futura.
    El corte de entrenamiento también excluye etiquetas de t+h aún no conocidas.
    """
    fechas = pd.Series(pd.to_datetime(np.asarray(fechas))).dt.to_period("M")
    out = []
    for anio in range(inicio, fin + 1, bloque_anios):
        comienzo = pd.Period(f"{anio}-01", freq="M")
        cierre = pd.Period(f"{min(anio + bloque_anios - 1, fin)}-12", freq="M")
        tr = np.flatnonzero((fechas + horizonte < comienzo).to_numpy())
        va = np.flatnonzero(((fechas >= comienzo) & (fechas <= cierre)).to_numpy())
        if len(tr) and len(va):
            out.append((tr, va))
    if not out:
        raise ValueError("No hay fechas suficientes para validar después del periodo de referencia.")
    return out


# ------------------------------------------------------------------ outliers
def limites_iqr(df, col, ref_mask, k=1.5):
    """Límites IQR por (punto, mes calendario) calculados SOLO con las filas ref_mask.
    Si el IQR es 0 (típico de meses secos: casi todo son ceros) se usa el percentil 99
    como límite superior para no marcar como 'outlier' cualquier llovizna."""
    g = df.loc[ref_mask].groupby(["punto", "mes"])[col]
    lim = pd.DataFrame({"q1": g.quantile(0.25), "q3": g.quantile(0.75), "p99": g.quantile(0.99)})
    iqr = lim["q3"] - lim["q1"]
    lim["lo"] = lim["q1"] - k * iqr
    lim["hi"] = np.where(iqr > 0, lim["q3"] + k * iqr, lim["p99"])
    return lim


def winsorizar_iqr(df, cols, ref_mask, k=1.5):
    out = df.copy()
    for c in cols:
        lim = limites_iqr(out, c, ref_mask, k)
        j = out[["punto", "mes"]].join(lim[["lo", "hi"]], on=["punto", "mes"])
        out[c] = out[c].clip(lower=j["lo"], upper=j["hi"])
    return out


def log_lluvia(df, cols=None):
    out = df.copy()
    for c in (cols or COLS_LLUVIA):
        out[c] = np.log1p(out[c])
    return out


def tratar_outliers(df, metodo, ref_mask=None):
    """metodo: 'ninguno' | 'log1p' | 'winsor'  (se aplica a las variables de lluvia)."""
    if metodo == "ninguno":
        return df.copy()
    if metodo == "log1p":
        return log_lluvia(df)
    if metodo == "winsor":
        if ref_mask is None:
            ref_mask = (df["fecha"] <= C.TRAIN_FIN)
        return winsorizar_iqr(df, COLS_LLUVIA, ref_mask)
    raise ValueError(metodo)

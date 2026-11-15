"""Pruebas rápidas (sin internet):  SEQUIA_SINTETICO=1 pytest -q"""
import os
os.environ["SEQUIA_SINTETICO"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src import config as C, datos, spi, features as F, modelos as M


def _mensual():
    d = datos.descargar_power_punto("Oruro", -17.97, -67.11, "19810101", "20201231")
    dia, rep = datos.limpiar_diario(d)
    return datos.a_mensual(dia), rep


def test_limpieza_convierte_999():
    m, rep = _mensual()
    assert rep["valores_-999"] > 0
    assert not (m["lluvia"] == -999).any()


def test_radiacion_power_en_mj_por_m2_dia():
    fechas = pd.date_range("2024-02-01", "2024-02-29", freq="D", name="fecha")
    valores = dict.fromkeys(C.PARAMS_POWER, 1.0)
    valores.update({"T2M": 10.0, "T2M_MAX": 18.0, "T2M_MIN": 2.0,
                    "RH2M": 50.0, "PS": 65.0, "ALLSKY_SFC_SW_DWN": 25.0})
    raw = pd.DataFrame(valores, index=fechas)
    raw["altitud"] = 3700.0
    raw["punto"] = "Oruro"
    raw.loc[fechas[0], "ALLSKY_SFC_SW_DWN"] = 55.0
    raw.loc[fechas[1], "ALLSKY_SFC_SW_DWN"] = -999.0

    limpio, rep = datos.limpiar_diario(raw)
    assert pd.isna(limpio.loc[fechas[0], "rad"])
    assert pd.isna(limpio.loc[fechas[1], "rad"])
    assert limpio.loc[fechas[2], "rad"] == 25.0
    assert rep["imposibles_a_NaN"]["rad"] == 1
    assert datos.a_mensual(limpio)["rad"].iloc[0] == 25.0


def test_spi_aprox_normal_en_referencia():
    m, _ = _mensual()
    p = spi.ajustar_parametros(m, 3)
    s = spi.calcular_spi(m, 3, p)
    ref = s[(m["fecha"].dt.year <= C.REF_FIN)].dropna()
    assert abs(ref.mean()) < 0.35 and 0.7 < ref.std() < 1.3


def test_split_temporal_sin_fuga():
    m, _ = _mensual()
    m2 = m.copy(); m2["punto"] = "Oruro"
    ps = {k: spi.ajustar_parametros(m2, k) for k in C.ESCALAS_SPI}
    ms = spi.agregar_spi(m2, ps)
    d = F.construir_features(ms, datos.descargar_oni(), F.calcular_climatologia(m2))
    f = F.lista_features()
    tr, va, te = F.particionar(d, 3, f)
    assert tr["fecha"].max() < va["fecha"].min() <= va["fecha"].max() < te["fecha"].min()
    assert (tr["fecha"] + pd.DateOffset(months=3)).max() <= pd.Timestamp(C.TRAIN_FIN) + pd.offsets.MonthEnd(0)


def test_cv_posterior_a_referencia_y_con_gap():
    fechas = pd.date_range("1981-01-01", "2018-12-01", freq="MS")
    cv = F.splits_post_referencia(fechas, horizonte=3)
    assert len(cv) == 4
    for tr, va in cv:
        assert fechas[va].min() >= pd.Timestamp("2011-01-01")
        assert (fechas[tr] + pd.DateOffset(months=3)).max() < fechas[va].min()


def test_climatologia_predice_puntos_conocidos_y_nuevos():
    x = pd.DataFrame({"punto": ["A", "A", "B", "B"], "mes": [1, 1, 1, 1]})
    y = pd.Series([1, 1, 0, 0])
    modelo = M.ClimatologiaClassifier(fuerza_prior=2).fit(x, y)
    p = modelo.predict_proba(pd.DataFrame({"punto": ["A", "B", "nuevo"], "mes": [1, 1, 1]}))
    assert p.shape == (3, 2)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[0, 1] > p[2, 1] > p[1, 1]
    assert np.isclose(p[2, 1], y.mean())


def _tabla_oni_html(formato):
    rng = np.random.default_rng(0)
    partes = ["<h1>Cold &amp; Warm Episodes</h1> PNS 26-05 published 2026 ", "Year DJF JFM FMA MAM AMJ MJJ JJA JAS ASO SON OND NDJ "]
    for anio in range(1950, 2027):
        n = 9 if anio == 2026 else 12
        vals = [round(float(v), 1) for v in rng.normal(0, 1, n)]
        if formato == "html":
            partes.append("<tr><td><b>%d</b></td>" % anio + "".join(f"<td><b>{v}</b></td>" if abs(v) >= .5 else f"<td>{v}</td>" for v in vals) + "</tr>")
        else:   # texto tipo markdown, como lo devuelve un navegador/lector
            partes.append(f"**{anio}** " + " ".join(f"**{v}**" if abs(v) >= .5 else f"{v}" for v in vals))
    return "\n".join(partes)


def test_parser_oni_html_y_texto():
    for fmt in ("html", "md"):
        df = datos._parsear_oni_html(_tabla_oni_html(fmt))
        assert len(df) == 76 * 12 + 9, (fmt, len(df))
        assert df["fecha"].min() == pd.Timestamp("1950-01-01") and df["fecha"].max() == pd.Timestamp("2026-09-01")


def test_parser_oni_txt_clasico():
    lineas = ["SEAS  YR  TOTAL  ANOM"]
    seas = datos.MESES_TRIM
    for anio in range(1950, 2026):
        for s_ in seas:
            lineas.append(f" {s_} {anio}  27.00  0.10")
    df = datos._parsear_oni_txt("\n".join(lineas))
    assert len(df) == 76 * 12 and df["fecha"].iloc[0] == pd.Timestamp("1950-01-01")

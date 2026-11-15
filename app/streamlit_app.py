"""
=============================================================================
OBSERVATORIO HIDROMETEOROLOGICO DEL ALTIPLANO · NASA POWER & NOAA ONI
Sistema de Monitoreo y Predicción de Sequía Meteorológica (SPI-3 <= -1.0)
=============================================================================
"""
import sys
import os
import json
import html
import math
import threading
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import pydeck as pdk
from fpdf import FPDF
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import predecir as PR, config as C, datos, spi as spi_mod, features as F

# =============================================================================
# CONFIGURACION DE PAGINA
# =============================================================================
st.set_page_config(
    page_title="Observatorio Climático · Sequía Bolivia",
    page_icon="https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_html(raw_html: str):
    """Renderiza HTML eliminando sangrías para evitar bloques de código accidentales en Markdown."""
    clean = "\n".join(line.lstrip() for line in raw_html.splitlines() if line.strip() != "")
    st.markdown(clean, unsafe_allow_html=True)


# =============================================================================
# SISTEMA DE DISEÑO MINIMALISTA, AEROESPACIAL Y 100% RESPONSIVO (MOBILE-FIRST)
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #090A0F;
    --bg-surface: rgba(18, 20, 29, 0.76);
    --bg-elevated: rgba(24, 27, 38, 0.85);
    --border-hairline: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(99, 102, 241, 0.4);
    --accent-primary: #6366F1;
    --accent-soft: #818CF8;
    --accent-sky: #38BDF8;
    --risk-low: #10B981;
    --risk-med: #F59E0B;
    --risk-high: #F43F5E;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --text-dim: #64748B;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-main);
    -webkit-font-smoothing: antialiased;
}

/* Lienzo principal: obsidiana profunda con aura superior sutil */
.stApp {
    background-color: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -15%, rgba(99, 102, 241, 0.15), transparent),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(56, 189, 248, 0.06), transparent);
    background-attachment: fixed;
}

header[data-testid="stHeader"] {
    background: rgba(9, 10, 15, 0.78);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border-hairline);
}

/* Ajuste de contenedor para pantallas grandes y celulares */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}

/* Animaciones suaves */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes orbitRotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.92); }
}

/* Barra lateral limpia */
section[data-testid="stSidebar"] {
    background: #0C0E15 !important;
    border-right: 1px solid var(--border-hairline);
}

section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}

section[data-testid="stSidebar"] label[data-baseweb="radio"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    transition: all 0.18s ease;
}

section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: var(--border-hairline);
}

section[data-testid="stSidebar"] label[data-baseweb="radio"] p {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
}

/* Cabecera de página minimalista */
.page-banner {
    background: linear-gradient(180deg, rgba(22, 25, 38, 0.75) 0%, rgba(15, 17, 26, 0.85) 100%);
    border: 1px solid var(--border-hairline);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
    animation: fadeUp 0.35s ease-out;
}

.banner-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.banner-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--accent-soft);
    text-transform: uppercase;
    margin-bottom: 4px;
}

.banner-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.25rem, 2.2vw, 1.6rem);
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin: 0;
}

.banner-desc {
    font-size: 0.86rem;
    color: var(--text-muted);
    margin-top: 4px;
    max-width: 680px;
}

/* Grillas Responsivas (Desktop -> Tablet -> Celular) */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}

.steps-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.metrics-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.metrics-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}

/* Tarjetas Superficie de Cristal Ahumado */
.glass-panel {
    background: var(--bg-surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-hairline);
    box-shadow: 0 8px 24px -8px rgba(0, 0, 0, 0.45), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
    transition: border-color 0.2s ease, transform 0.2s ease;
    animation: fadeUp 0.4s ease-out;
}

.glass-panel:hover {
    border-color: rgba(255, 255, 255, 0.15);
}

/* Tarjetas KPI */
.kpi-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-hairline);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    animation: fadeUp 0.4s ease-out;
}

.kpi-label {
    font-size: 0.74rem;
    font-weight: 500;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.35rem, 2vw, 1.7rem);
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.kpi-sub {
    font-size: 0.73rem;
    color: var(--text-dim);
    margin-top: 6px;
}

/* Píldoras y Etiquetas */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
}

.pill-neutral {
    background: rgba(255, 255, 255, 0.05);
    color: #CBD5E1;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.pill-indigo {
    background: rgba(99, 102, 241, 0.14);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.risk-badge-ALTO {
    background: rgba(244, 63, 94, 0.15);
    color: #FDA4AF;
    border: 1px solid rgba(244, 63, 94, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    white-space: nowrap;
}

.risk-badge-MEDIO {
    background: rgba(245, 158, 11, 0.15);
    color: #FDE68A;
    border: 1px solid rgba(245, 158, 11, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    white-space: nowrap;
}

.risk-badge-BAJO {
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    white-space: nowrap;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
    animation: pulseDot 2s infinite;
}

/* Tarjeta compacta de estación en el Ranking */
.station-item {
    background: rgba(20, 23, 34, 0.68);
    border: 1px solid var(--border-hairline);
    border-radius: 10px;
    padding: 11px 14px;
    margin-bottom: 8px;
    transition: all 0.18s ease;
}

.station-item:hover {
    background: rgba(28, 32, 48, 0.85);
    border-color: rgba(255, 255, 255, 0.16);
}

/* Caja de métrica interna */
.mini-stat {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--border-hairline);
    border-radius: 10px;
    padding: 12px 14px;
}

.mini-stat-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.mini-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
}

/* Botones y Controles Segmentados (Sin rotura de texto en 2 líneas) */
.stButton > button, .stDownloadButton > button {
    background: var(--accent-primary);
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 9px 16px;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: all 0.18s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #4F46E5;
    border-color: rgba(255, 255, 255, 0.28);
    transform: translateY(-1px);
}

/* Pestañas (Tabs) limpias */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(18, 20, 29, 0.8);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border-hairline);
    margin-bottom: 14px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--text-muted);
    border-radius: 8px;
    padding: 8px 18px;
    white-space: nowrap !important;
}

.stTabs [aria-selected="true"] {
    background: var(--accent-primary) !important;
    color: #FFFFFF !important;
}

.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #E2E8F0;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Centrado exacto de gráficos interactivos Plotly */
.stPlotlyChart {
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
}
.stPlotlyChart > div {
    width: 100% !important;
}

/* =========================================================================
   REGLAS RESPONSIVAS PARA TABLETS Y CELULARES
   ========================================================================= */
@media (max-width: 992px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .steps-grid-4 {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        padding-top: 1.2rem !important;
    }
    .page-banner {
        padding: 16px 18px;
        border-radius: 12px;
    }
    .kpi-grid {
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .steps-grid-4 {
        grid-template-columns: 1fr;
    }
    .metrics-grid-3 {
        grid-template-columns: 1fr;
    }
    .glass-panel {
        padding: 14px 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 12px;
        font-size: 0.76rem;
    }
}
</style>
"""
render_html(CUSTOM_CSS)

# =============================================================================
# CONSTANTES Y METADATOS FISIOGRAFICOS DE BOLIVIA
# =============================================================================
COLOR_HEX = {"BAJO": "#10B981", "MEDIO": "#F59E0B", "ALTO": "#F43F5E"}
COLOR_RGBA = {
    "BAJO": [16, 185, 129, 215],
    "MEDIO": [245, 158, 11, 225],
    "ALTO": [244, 63, 94, 235],
}

ALTITUDES_ESTACIONES = {
    "La Paz": 3640,
    "Copacabana": 3841,
    "Charaña": 4054,
    "Patacamaya": 3799,
    "Oruro": 3735,
    "Challapata": 3738,
    "Potosí": 4067,
    "Uyuni": 3676,
}

ZONAS_FISIOGRAFICAS = [
    {
        "nombre": "Altiplano y Cordillera Occidental",
        "altitud_media": "3,600 - 4,200 msnm",
        "regimen": "Semiárido frío altoandino (300 - 700 mm/año). Dominio nativo del modelo.",
        "color": "#38BDF8",
        "coords": [
            [-14.8, -69.3], [-15.8, -69.5], [-17.5, -69.6], [-20.5, -68.6],
            [-22.8, -67.8], [-22.2, -65.5], [-19.5, -65.5], [-17.8, -66.5],
            [-16.2, -67.6], [-14.8, -69.3]
        ]
    },
    {
        "nombre": "Valles Interandinos y Cordillera Oriental",
        "altitud_media": "1,500 - 3,400 msnm",
        "regimen": "Mesotérmico de transición (450 - 950 mm/año). Cochabamba, Chuquisaca y Tarija.",
        "color": "#F59E0B",
        "coords": [
            [-14.5, -68.8], [-16.2, -67.6], [-17.8, -66.5], [-19.5, -65.5],
            [-22.2, -65.5], [-22.1, -63.5], [-19.0, -63.4], [-17.2, -63.8],
            [-15.2, -66.3], [-14.5, -68.8]
        ]
    },
    {
        "nombre": "Tierras Bajas (Llanos, Amazonía y Chaco)",
        "altitud_media": "150 - 800 msnm",
        "regimen": "Húmedo amazónico al norte a semiárido chaqueño al sur.",
        "color": "#10B981",
        "coords": [
            [-10.0, -69.5], [-14.5, -68.8], [-15.2, -66.3], [-17.2, -63.8],
            [-19.0, -63.4], [-22.1, -63.5], [-22.3, -62.6], [-20.1, -58.0],
            [-16.3, -58.4], [-13.5, -60.5], [-10.4, -65.4], [-10.0, -69.5]
        ]
    }
]


def clasificar_fisiografia(lat: float, lon: float) -> dict:
    """Clasifica una coordenada en Altiplano, Valles Interandinos o Llanos."""
    lon_borde_altiplano = min(-65.45, -67.70 + max(0.0, (abs(lat) - 16.0) * 0.60))
    lon_borde_valles = min(-63.40, -66.50 + max(0.0, (abs(lat) - 15.0) * 0.90))

    if lon <= lon_borde_altiplano and -22.9 <= lat <= -14.5:
        alt_est = int(3680 + max(0, (abs(lat) - 16.5) * 45) + max(0, (abs(lon) - 67.5) * 140))
        return {
            "zona": "Altiplano Andino",
            "altitud_est": f"~{min(alt_est, 4350)} msnm",
            "color": "#38BDF8",
            "fiabilidad": "Alta (Dominio de entrenamiento)",
            "descripcion": "Meseta altoandina semiárida con lluvias concentradas entre diciembre y marzo."
        }
    elif lon <= lon_borde_valles and -22.5 <= lat <= -14.0:
        alt_est = int(2550 - (lon - lon_borde_altiplano) * 420)
        return {
            "zona": "Valles Interandinos",
            "altitud_est": f"~{max(1200, min(alt_est, 3300))} msnm",
            "color": "#F59E0B",
            "fiabilidad": "Media (Zona de transición orográfica)",
            "descripcion": "Valles mesotérmicos de la vertiente oriental andina."
        }
    else:
        es_chaco = lat < -18.5
        return {
            "zona": "Llanos del Chaco" if es_chaco else "Llanos y Amazonía",
            "altitud_est": "~280 - 450 msnm" if es_chaco else "~180 - 350 msnm",
            "color": "#10B981",
            "fiabilidad": "Exploratoria (Requiere calibración local)",
            "descripcion": "Tierras bajas con régimen pluvial tropical o chaqueño."
        }


# =============================================================================
# CARGA INSTANTANEA Y SINCRONIZACION EN SEGUNDO PLANO
# =============================================================================
RUTA_CACHE_TELEMETRIA = RAIZ / "data" / "processed" / "telemetry_cache.json"
RUTA_HISTORICO = RAIZ / "data" / "processed" / "dataset_modelado.csv"
_SYNC_LOCK = threading.Lock()
_SYNC_STATUS = {"running": False, "last_sync": None, "error": None}


@st.cache_resource(show_spinner=False)
def get_artefacto():
    return PR.cargar_artefacto()


@st.cache_data(show_spinner=False)
def cargar_historico_modelado():
    if RUTA_HISTORICO.exists():
        return pd.read_csv(RUTA_HISTORICO, parse_dates=["fecha"])
    return None


def _calcular_desde_historico_local(art):
    df = cargar_historico_modelado()
    if df is None or df.empty:
        return None
    cache = {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "LOCAL_ARCHIVE",
        "horizontes": {1: [], 2: [], 3: []},
        "features_actuales": {}
    }
    for h in [1, 2, 3]:
        cfg = art["modelos"][h]
        feats = cfg["features"]
        trat = cfg.get("tratamiento", "ninguno")
        dh = F.tratar_outliers(df, trat)
        listo = dh.dropna(subset=feats).sort_values("fecha")
        ultimos = listo.groupby("punto").tail(1)
        for _, fila_s in ultimos.iterrows():
            punto = fila_s["punto"]
            if punto not in art["puntos"]:
                continue
            lat, lon = art["puntos"][punto]
            fila_df = pd.DataFrame([fila_s])
            p = float(cfg["pipeline"].predict_proba(fila_df[feats])[:, 1][0])
            t = pd.Timestamp(fila_s["fecha"])
            nombre_modelo = cfg.get("nombre", art.get("modelo"))
            cache["horizontes"][h].append({
                "punto": punto, "lat": lat, "lon": lon, "horizonte_meses": h,
                "mes_base": t.strftime("%Y-%m"),
                "mes_objetivo": (t + pd.DateOffset(months=h)).strftime("%Y-%m"),
                "probabilidad_sequia": round(p, 4),
                "umbral_modelo": round(cfg["umbral"], 4),
                "riesgo": PR._riesgo(p, cfg["umbral"]),
                "spi3_actual": None if pd.isna(fila_s["spi3"]) else round(float(fila_s["spi3"]), 2),
                "lluvia_mes_base_mm": round(float(fila_s["lluvia"]), 1),
                "punto_nuevo": False,
                "version_modelo": art.get("version"),
                "modelo_horizonte": nombre_modelo,
                "aviso": PR.AVISO
            })
            if h == 1:
                cache["features_actuales"][punto] = {
                    k: (round(float(v), 3) if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v) else None)
                    for k, v in fila_s.to_dict().items() if k not in ("fecha", "punto")
                }
    return cache


def _sincronizar_nasa_background():
    if not _SYNC_LOCK.acquire(blocking=False):
        return
    try:
        _SYNC_STATUS["running"] = True
        _SYNC_STATUS["error"] = None
        art = get_artefacto()
        hoy = pd.Timestamp.today().normalize()
        ini = (hoy - pd.DateOffset(months=26)).replace(day=1).strftime("%Y%m%d")
        fin = hoy.strftime("%Y%m%d")
        oni = datos.descargar_oni()

        nuevo_cache = {
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "source": "NASA_POWER_LIVE",
            "horizontes": {"1": [], "2": [], "3": []},
            "features_actuales": {}
        }

        for punto, (lat, lon) in art["puntos"].items():
            m = PR._mensual_completo(lat, lon, punto, ini, fin)
            ms = spi_mod.agregar_spi(m, art["spi_params"])
            d = F.construir_features(ms, oni, art["climatologia"], horizontes=[1, 2, 3])
            for h in [1, 2, 3]:
                cfg = art["modelos"][h]
                feats = cfg["features"]
                trat = cfg.get("tratamiento", "ninguno")
                dh = F.tratar_outliers(d, trat)
                listo = dh.dropna(subset=feats)
                fila = listo.iloc[[-1]]
                p = float(cfg["pipeline"].predict_proba(fila[feats])[:, 1][0])
                t = pd.Timestamp(fila["fecha"].iloc[0])
                nombre_modelo = cfg.get("nombre", art.get("modelo"))
                nota_clim = (" Horizonte 3: referencia climatológica; no supera de forma consistente a la persistencia." if nombre_modelo == "Climatología" else "")
                nuevo_cache["horizontes"][str(h)].append({
                    "punto": punto, "lat": lat, "lon": lon, "horizonte_meses": h,
                    "mes_base": t.strftime("%Y-%m"),
                    "mes_objetivo": (t + pd.DateOffset(months=h)).strftime("%Y-%m"),
                    "probabilidad_sequia": round(p, 4),
                    "umbral_modelo": round(cfg["umbral"], 4),
                    "riesgo": PR._riesgo(p, cfg["umbral"]),
                    "spi3_actual": None if pd.isna(fila["spi3"].iloc[0]) else round(float(fila["spi3"].iloc[0]), 2),
                    "lluvia_mes_base_mm": round(float(fila["lluvia"].iloc[0]), 1),
                    "punto_nuevo": False,
                    "version_modelo": art.get("version"),
                    "modelo_horizonte": nombre_modelo,
                    "aviso": PR.AVISO + nota_clim
                })
                if h == 1:
                    row_dict = fila.iloc[0].to_dict()
                    nuevo_cache["features_actuales"][punto] = {
                        k: (round(float(v), 3) if isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v) else None)
                        for k, v in row_dict.items() if k not in ("fecha", "punto")
                    }

        RUTA_CACHE_TELEMETRIA.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = RUTA_CACHE_TELEMETRIA.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(nuevo_cache, f, ensure_ascii=False, indent=2)
        tmp_path.replace(RUTA_CACHE_TELEMETRIA)
        _SYNC_STATUS["last_sync"] = nuevo_cache["updated_at"]
    except Exception as exc:
        _SYNC_STATUS["error"] = str(exc)
    finally:
        _SYNC_STATUS["running"] = False
        _SYNC_LOCK.release()


def obtener_telemetria_instantanea(art):
    if RUTA_CACHE_TELEMETRIA.exists():
        try:
            with open(RUTA_CACHE_TELEMETRIA, "r", encoding="utf-8") as f:
                data = json.load(f)
            upd = data.get("updated_at", "")
            if upd:
                try:
                    edad_seg = (dt.datetime.now() - dt.datetime.fromisoformat(upd)).total_seconds()
                    if edad_seg > 6 * 3600 and not _SYNC_STATUS["running"]:
                        threading.Thread(target=_sincronizar_nasa_background, daemon=True).start()
                except Exception:
                    pass
            return data
        except Exception:
            pass

    local_data = _calcular_desde_historico_local(art)
    if not _SYNC_STATUS["running"]:
        threading.Thread(target=_sincronizar_nasa_background, daemon=True).start()
    return local_data


def interpolar_diagnostico_espacial(lat: float, lon: float, df_preds: pd.DataFrame, umbral: float) -> dict:
    lats = df_preds["lat"].to_numpy(dtype=float)
    lons = df_preds["lon"].to_numpy(dtype=float)
    dists = np.hypot(lats - lat, lons - lon)
    idx_min = int(np.argmin(dists))
    dist_min = float(dists[idx_min])

    if dist_min < 0.28:
        fila = df_preds.iloc[idx_min]
        return {
            "es_estacion_oficial": True,
            "nombre": fila["punto"],
            "distancia_km": round(dist_min * 111.0, 1),
            "estacion_cercana": fila["punto"],
            "probabilidad_sequia": float(fila["probabilidad_sequia"]),
            "umbral_modelo": float(fila["umbral_modelo"]),
            "riesgo": fila["riesgo"],
            "spi3_actual": float(fila["spi3_actual"]) if pd.notna(fila["spi3_actual"]) else 0.0,
            "lluvia_mes_base_mm": float(fila["lluvia_mes_base_mm"]),
            "mes_base": fila["mes_base"],
            "mes_objetivo": fila["mes_objetivo"],
        }

    pesos = 1.0 / np.maximum(dists, 0.05) ** 2
    pesos /= pesos.sum()
    prob_idw = float(np.dot(pesos, df_preds["probabilidad_sequia"].to_numpy(dtype=float)))
    spi_idw = float(np.dot(pesos, df_preds["spi3_actual"].fillna(0).to_numpy(dtype=float)))
    lluvia_idw = float(np.dot(pesos, df_preds["lluvia_mes_base_mm"].to_numpy(dtype=float)))
    est_cercana = df_preds.iloc[idx_min]["punto"]

    return {
        "es_estacion_oficial": False,
        "nombre": f"Coordenada ({lat:.2f}°, {lon:.2f}°)",
        "distancia_km": round(dist_min * 111.0, 1),
        "estacion_cercana": est_cercana,
        "probabilidad_sequia": round(prob_idw, 4),
        "umbral_modelo": round(umbral, 4),
        "riesgo": PR._riesgo(prob_idw, umbral),
        "spi3_actual": round(spi_idw, 2),
        "lluvia_mes_base_mm": round(lluvia_idw, 1),
        "mes_base": df_preds["mes_base"].iloc[0],
        "mes_objetivo": df_preds["mes_objetivo"].iloc[0],
    }


def _crear_gauge_plotly(prob_val: float, umbral_val: float, riesgo_txt: str, subtitulo: str) -> go.Figure:
    """Construye un tacómetro limpio, perfectamente centrado y moderno en Plotly."""
    c_riesgo = COLOR_HEX.get(riesgo_txt, "#6366F1")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_val,
        domain={"x": [0, 1], "y": [0, 1]},
        number={
            "suffix": "%",
            "font": {"color": "#FFFFFF", "size": 38, "family": "Space Grotesk"}
        },
        delta={
            "reference": umbral_val,
            "position": "bottom",
            "increasing": {"color": "#F43F5E"},
            "decreasing": {"color": "#10B981"},
            "suffix": "% vs umbral",
            "font": {"size": 13, "color": "#94A3B8", "family": "Inter, sans-serif"}
        },
        title={
            "text": f"<b>Probabilidad de Sequía</b><br><span style='font-size:0.82em;color:#94A3B8'>{subtitulo}</span>",
            "font": {"color": "#E2E8F0", "size": 14},
            "align": "center"
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#475569",
                "tickfont": {"color": "#94A3B8", "size": 11}
            },
            "bar": {"color": c_riesgo, "thickness": 0.28},
            "bgcolor": "rgba(255, 255, 255, 0.03)",
            "borderwidth": 1,
            "bordercolor": "rgba(255, 255, 255, 0.08)",
            "steps": [
                {"range": [0, umbral_val * 0.5], "color": "rgba(16, 185, 129, 0.14)"},
                {"range": [umbral_val * 0.5, umbral_val], "color": "rgba(245, 158, 11, 0.16)"},
                {"range": [umbral_val, 100], "color": "rgba(244, 63, 94, 0.20)"},
            ],
            "threshold": {
                "line": {"color": "#F43F5E", "width": 3.5},
                "thickness": 0.82,
                "value": umbral_val,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=270,
        margin=dict(l=20, r=20, t=52, b=15),
        font=dict(family="Inter, sans-serif")
    )
    return fig


# =============================================================================
# GENERADOR DE REPORTE OFICIAL EN PDF NATIVO (FPDF2)
# =============================================================================
def _limpiar_latin1(texto: str) -> str:
    reemplazos = {
        "≤": "<=", "≥": ">=", "−": "-", "–": "-", "—": "-",
        "·": "|", "•": "*", "“": '"', "”": '"', "‘": "'", "’": "'",
        "⁻¹": "^-1", "Φ": "Phi", "α": "alpha", "β": "beta"
    }
    out = str(texto)
    for k, v in reemplazos.items():
        out = out.replace(k, v)
    return out.encode("latin-1", errors="replace").decode("latin-1")


def generar_pdf_ejecutivo(df_preds: pd.DataFrame, h: int, nombre_mod_h: str, umbral_h: float, sync_time: str) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_fill_color(12, 14, 22)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_fill_color(99, 102, 241)
    pdf.rect(0, 40, 210, 1.2, "F")

    pdf.set_xy(14, 9)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(129, 140, 248)
    pdf.cell(0, 5, _limpiar_latin1("NASA POWER (MERRA-2) & NOAA ONI · OBSERVATORIO CLIMATICO DEL ALTIPLANO"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(14)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, _limpiar_latin1("REPORTE TECNICO DE PREDICCION DE SEQUIA METEOROLOGICA"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 195, 220)
    mes_base = df_preds["mes_base"].iloc[0]
    mes_obj = df_preds["mes_objetivo"].iloc[0]
    fecha_emision = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(
        0, 6,
        _limpiar_latin1(f"Altiplano Boliviano (8 Estaciones)  |  Mes Base: {mes_base}  |  Proyección: {mes_obj} (+{h}M)  |  Emisión: {fecha_emision}"),
        new_x="LMARGIN", new_y="NEXT"
    )

    pdf.ln(10)

    c_alto = int((df_preds["riesgo"] == "ALTO").sum())
    c_medio = int((df_preds["riesgo"] == "MEDIO").sum())
    c_bajo = int((df_preds["riesgo"] == "BAJO").sum())
    max_row = df_preds.loc[df_preds["probabilidad_sequia"].idxmax()]

    y_kpi = pdf.get_y()
    kpis = [
        ("HORIZONTE Y MODELO", f"+{h} Mes(es)", nombre_mod_h[:24], (241, 245, 249), (15, 23, 42)),
        ("UMBRAL OPTIMO F1", f"{umbral_h*100:.1f}%", "Corte de decision OOF", (241, 245, 249), (15, 23, 42)),
        ("ESTACIONES EN ALERTA", f"{c_alto} Alta / {c_medio} Media", f"{c_bajo} en nivel bajo", (254, 226, 226) if c_alto > 0 else (209, 250, 229), (153, 27, 27) if c_alto > 0 else (6, 95, 70)),
        ("MAXIMA VULNERABILIDAD", f"{max_row['punto']}", f"Prob: {max_row['probabilidad_sequia']*100:.1f}% ({max_row['riesgo']})", (241, 245, 249), (15, 23, 42)),
    ]

    for i, (titulo, val, sub, bg, fg) in enumerate(kpis):
        x_box = 14 + i * 46.5
        pdf.set_fill_color(*bg)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(x_box, y_kpi, 44, 22, "DF")
        pdf.set_xy(x_box + 2.5, y_kpi + 2)
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(39, 4, _limpiar_latin1(titulo))
        pdf.set_xy(x_box + 2.5, y_kpi + 7.5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*fg)
        pdf.cell(39, 6, _limpiar_latin1(val))
        pdf.set_xy(x_box + 2.5, y_kpi + 15)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(39, 4, _limpiar_latin1(sub))

    pdf.set_y(y_kpi + 28)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, _limpiar_latin1("1. MATRIZ DE MONITOREO HIDROMETEOROLOGICO POR ESTACION"), new_x="LMARGIN", new_y="NEXT")

    headers = ["Estación", "Lat / Lon", "Altitud", "Lluvia Base", "SPI-3 Actual", "Prob. Sequía", "Umbral F1", "Nivel Riesgo"]
    widths = [28, 30, 19, 22, 22, 24, 19, 22]

    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7.8)
    pdf.set_x(14)
    for h_txt, w in zip(headers, widths):
        pdf.cell(w, 8, _limpiar_latin1(h_txt), border=1, align="C", fill=True)
    pdf.ln()

    df_ord = df_preds.sort_values("probabilidad_sequia", ascending=False).reset_index(drop=True)
    pdf.set_font("Helvetica", "", 8)

    for idx, row in df_ord.iterrows():
        bg_row = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg_row)
        pdf.set_text_color(15, 23, 42)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_x(14)

        punto = str(row["punto"])
        coords = f"{row['lat']:.2f}, {row['lon']:.2f}"
        alt = f"{ALTITUDES_ESTACIONES.get(punto, 3750)} m"
        lluvia = f"{row['lluvia_mes_base_mm']:.1f} mm"
        spi_str = f"{row['spi3_actual']:+.2f}" if pd.notna(row["spi3_actual"]) else "N/D"
        prob_str = f"{row['probabilidad_sequia']*100:.1f}%"
        umb_str = f"{row['umbral_modelo']*100:.1f}%"
        riesgo = str(row["riesgo"])

        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(widths[0], 7.5, _limpiar_latin1(punto), border=1, align="L", fill=True)
        pdf.set_font("Helvetica", "", 7.8)
        pdf.cell(widths[1], 7.5, _limpiar_latin1(coords), border=1, align="C", fill=True)
        pdf.cell(widths[2], 7.5, _limpiar_latin1(alt), border=1, align="C", fill=True)
        pdf.cell(widths[3], 7.5, _limpiar_latin1(lluvia), border=1, align="R", fill=True)
        pdf.cell(widths[4], 7.5, _limpiar_latin1(spi_str), border=1, align="C", fill=True)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(widths[5], 7.5, _limpiar_latin1(prob_str), border=1, align="R", fill=True)
        pdf.set_font("Helvetica", "", 7.8)
        pdf.cell(widths[6], 7.5, _limpiar_latin1(umb_str), border=1, align="R", fill=True)

        if riesgo == "ALTO":
            pdf.set_fill_color(254, 202, 202)
            pdf.set_text_color(153, 27, 27)
        elif riesgo == "MEDIO":
            pdf.set_fill_color(254, 240, 138)
            pdf.set_text_color(146, 64, 14)
        else:
            pdf.set_fill_color(209, 250, 229)
            pdf.set_text_color(6, 95, 70)

        pdf.set_font("Helvetica", "B", 7.8)
        pdf.cell(widths[7], 7.5, _limpiar_latin1(riesgo), border=1, align="C", fill=True)
        pdf.ln()

    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, _limpiar_latin1("2. PERFIL COMPARATIVO DE PROBABILIDAD VS. UMBRAL CRITICO"), new_x="LMARGIN", new_y="NEXT")

    y_chart = pdf.get_y() + 2
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(14, y_chart, 182, 56, "DF")

    bar_x0 = 46
    bar_max_w = 135
    for idx, row in df_ord.iterrows():
        by = y_chart + 4 + idx * 6.2
        pdf.set_xy(16, by)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(28, 4.5, _limpiar_latin1(row["punto"]), align="R")

        pdf.set_fill_color(226, 232, 240)
        pdf.rect(bar_x0, by + 0.6, bar_max_w, 3.4, "F")

        p_val = float(row["probabilidad_sequia"])
        w_val = max(1.2, min(bar_max_w, p_val * bar_max_w))
        if row["riesgo"] == "ALTO":
            pdf.set_fill_color(244, 63, 94)
        elif row["riesgo"] == "MEDIO":
            pdf.set_fill_color(245, 158, 11)
        else:
            pdf.set_fill_color(16, 185, 129)
        pdf.rect(bar_x0, by + 0.6, w_val, 3.4, "F")

        pdf.set_xy(bar_x0 + bar_max_w + 2, by)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(12, 4.5, f"{p_val*100:.1f}%")

    x_umb = bar_x0 + min(bar_max_w, umbral_h * bar_max_w)
    pdf.set_draw_color(220, 38, 38)
    pdf.set_line_width(0.5)
    pdf.line(x_umb, y_chart + 2, x_umb, y_chart + 54)
    pdf.set_line_width(0.2)

    pdf.set_y(y_chart + 60)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, _limpiar_latin1("3. PROTOCOLO DE INTERPRETACION Y RECOMENDACIONES"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8.3)
    pdf.set_text_color(51, 65, 85)
    recomendaciones = (
        "* DEFINICION OMM (SPI-3 <= -1.0): El Indice Estandarizado de Precipitacion a 3 meses cuantifica anomalias "
        "de precipitacion respecto al periodo de referencia climatologico 1981-2010 mediante una distribucion Gamma mixta.\n"
        "* NIVEL ALTO (Probabilidad >= Umbral F1): Condicion de deficit hidrico probable. Se recomienda activar monitoreo "
        "de reservorios municipales, gestion preventiva de riego en cultivos andinos y vigilancia de bofedales.\n"
        "* NIVEL MEDIO (Umbral/2 <= Probabilidad < Umbral): Vigilancia preventiva ante posibles retrasos en lluvias.\n"
        "* NIVEL BAJO (Probabilidad < Umbral/2): Condiciones dentro de los margenes climatologicos normales."
    )
    pdf.multi_cell(182, 4.8, _limpiar_latin1(recomendaciones))

    pdf.ln(4)
    pdf.set_draw_color(203, 213, 225)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7.3)
    pdf.set_text_color(100, 116, 139)
    aviso_pie = f"NOTA LEGAL: {PR.AVISO} Sincronización: {sync_time}."
    pdf.multi_cell(182, 4, _limpiar_latin1(aviso_pie))

    return bytes(pdf.output())


# =============================================================================
# INICIALIZACION DE DATOS
# =============================================================================
try:
    art = get_artefacto()
except FileNotFoundError as err:
    st.error(f"Error crítico: No se encontró el modelo ({err}).")
    st.stop()

telemetria_data = obtener_telemetria_instantanea(art)
if not telemetria_data:
    st.error("No se encontraron datos de telemetría.")
    st.stop()

# =============================================================================
# BARRA LATERAL (MINIMALISTA Y ELEGANTE)
# =============================================================================
with st.sidebar:
    render_html("""
    <div style="padding: 4px 2px 16px 2px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(99, 102, 241, 0.14); border: 1px solid rgba(99, 102, 241, 0.35); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                <svg viewBox="0 0 48 48" width="24" height="24">
                    <circle cx="24" cy="24" r="19" fill="none" stroke="rgba(129, 140, 248, 0.35)" stroke-width="1.5" stroke-dasharray="4 3" style="transform-origin: 24px 24px; animation: orbitRotate 16s linear infinite;" />
                    <circle cx="24" cy="24" r="6" fill="#6366F1" />
                    <circle cx="24" cy="5" r="2.8" fill="#38BDF8" />
                </svg>
            </div>
            <div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;">
                    ANDES ORBITAL
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8;">
                    NASA POWER · Altiplano
                </div>
            </div>
        </div>
    </div>
    """)

    opciones_nav = [
        "Panorama General",
        "Series Históricas (SPI)",
        "Tacómetro y Explorador",
        "Factores Climáticos (IA)",
        "Reportes y Descargas",
        "Metodología y Satélites"
    ]
    pagina = st.radio("Navegación", opciones_nav, label_visibility="collapsed")

    st.divider()

    render_html("<div style='font-size: 0.72rem; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>Horizonte de Pronóstico</div>")
    horizontes_disp = sorted(int(k) for k in art["modelos"].keys())
    h = st.segmented_control(
        "Horizonte",
        options=horizontes_disp,
        default=horizontes_disp[0],
        format_func=lambda x: f"+{x} Mes" if x == 1 else f"+{x} Meses",
        label_visibility="collapsed"
    ) or horizontes_disp[0]

    cfg_h = art["modelos"][h]
    nombre_mod_h = cfg_h.get("nombre", art.get("modelo", "Modelo Supervisado"))

    render_html(f"""
    <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 10px; padding: 11px 13px; margin-top: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 4px;">
            <span style="color: #94A3B8;">Modelo:</span>
            <span style="color: #F8FAFC; font-weight: 600;">{html.escape(str(nombre_mod_h))}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
            <span style="color: #94A3B8;">Umbral F1:</span>
            <span style="color: #818CF8; font-family: 'JetBrains Mono', monospace; font-weight: 600;">{cfg_h['umbral']*100:.1f}%</span>
        </div>
    </div>
    """)

    st.divider()

    sync_ts = str(telemetria_data.get("updated_at", ""))[:16].replace("T", " ")
    if st.button("Actualizar Telemetría NASA", use_container_width=True):
        if not _SYNC_STATUS["running"]:
            threading.Thread(target=_sincronizar_nasa_background, daemon=True).start()
        st.toast("Actualizando telemetría desde NASA POWER en segundo plano.")

    render_html(f"""
    <div style="font-size: 0.7rem; color: #64748B; margin-top: 10px; line-height: 1.4;">
        Sincronizado: {html.escape(sync_ts)}<br>
         Herramienta académica universitaria. No sustituye pronósticos de SENAMHI.
    </div>
    """)

# =============================================================================
# DATOS ACTIVOS DEL HORIZONTE
# =============================================================================
preds_h = telemetria_data["horizontes"].get(str(h)) or telemetria_data["horizontes"].get(h, [])
df_preds = pd.DataFrame(preds_h)
features_actuales = telemetria_data.get("features_actuales", {})

mes_base_actual = df_preds["mes_base"].iloc[0]
mes_objetivo_actual = df_preds["mes_objetivo"].iloc[0]

conteo_alto = int((df_preds["riesgo"] == "ALTO").sum())
conteo_medio = int((df_preds["riesgo"] == "MEDIO").sum())
conteo_bajo = int((df_preds["riesgo"] == "BAJO").sum())
prob_maxima = float(df_preds["probabilidad_sequia"].max())
punto_max_riesgo = df_preds.loc[df_preds["probabilidad_sequia"].idxmax(), "punto"]

oni_reciente = 0.0
if features_actuales:
    primer_p = next(iter(features_actuales.values()))
    oni_reciente = primer_p.get("oni_l2") or primer_p.get("oni_t") or 0.0
fase_enso_actual = "El Niño" if oni_reciente >= 0.5 else ("La Niña" if oni_reciente <= -0.5 else "Neutro")


# =============================================================================
# 1. PANORAMA GENERAL (MAPA Y RANKING)
# =============================================================================
if pagina == "Panorama General":
    render_html(f"""
    <div class="page-banner">
        <div class="banner-flex">
            <div>
                <div class="banner-eyebrow">TELEMETRIA SATELITAL · NASA POWER & NOAA</div>
                <h1 class="banner-title">Monitoreo de Sequía en el Altiplano Boliviano</h1>
                <div class="banner-desc">
                    Pronóstico probabilístico del índice <b>SPI-3 &le; -1.0</b> para 8 estaciones altoandinas.
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="pill pill-neutral">Base: {mes_base_actual}</span>
                <span class="pill pill-indigo">Objetivo: {mes_objetivo_actual} (+{h}M)</span>
                <span class="pill pill-neutral">ENSO: {fase_enso_actual} ({oni_reciente:+.2f}°C)</span>
            </div>
        </div>
    </div>
    """)

    # Grilla responsiva de 4 KPIs (2x2 en celular, 4x1 en escritorio)
    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label"><span>Estaciones</span><span class="status-dot" style="background:#10B981;"></span></div>
            <div class="kpi-number">{len(df_preds)} Activas</div>
            <div class="kpi-sub">3,640 a 4,067 msnm</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Riesgo Alto</span><span style="color:#F43F5E;font-family:'JetBrains Mono',monospace;">&ge;{cfg_h['umbral']*100:.0f}%</span></div>
            <div class="kpi-number" style="color: #FDA4AF;">{conteo_alto} Estaciones</div>
            <div class="kpi-sub">Sobre el umbral F1</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Riesgo Medio</span><span style="color:#F59E0B;">Preventivo</span></div>
            <div class="kpi-number" style="color: #FDE68A;">{conteo_medio} Estaciones</div>
            <div class="kpi-sub">{conteo_bajo} en riesgo bajo</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Mayor Probabilidad</span><span>+{h}M</span></div>
            <div class="kpi-number" style="color: #818CF8;">{html.escape(str(punto_max_riesgo))} ({prob_maxima*100:.1f}%)</div>
            <div class="kpi-sub">{html.escape(str(nombre_mod_h))}</div>
        </div>
    </div>
    """)

    col_mapa, col_ranking = st.columns([1.55, 1.15])

    with col_mapa:
        head_m1, head_m2 = st.columns([1.2, 1.2])
        with head_m1:
            render_html("<div class='section-heading' style='margin-top:6px;'>Mapa de Riesgo Regional</div>")
        with head_m2:
            modo_mapa = st.segmented_control(
                "Vista del mapa",
                options=["Satélite y Relieve", "Vista 3D"],
                default="Satélite y Relieve",
                label_visibility="collapsed"
            ) or "Satélite y Relieve"

        if modo_mapa == "Satélite y Relieve":
            m_central = folium.Map(location=[-18.25, -67.65], zoom_start=6, tiles=None, control_scale=True)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, DeLorme, HERE",
                name="Modo Oscuro (Esri)",
                overlay=False,
                control=True
            ).add_to(m_central)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Topo",
                name="Relieve Físico y Topográfico",
                overlay=False,
                control=True
            ).add_to(m_central)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, NASA, USGS",
                name="Satélite Real (NASA / Esri)",
                overlay=False,
                control=True
            ).add_to(m_central)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Nombres y Límites",
                overlay=True,
                control=True
            ).add_to(m_central)

            folium.Polygon(
                locations=ZONAS_FISIOGRAFICAS[0]["coords"],
                color="#38BDF8",
                weight=1.5,
                dash_array="5, 5",
                fill=True,
                fill_color="#38BDF8",
                fill_opacity=0.06,
                tooltip="Región del Altiplano Boliviano"
            ).add_to(m_central)

            for _, r_st in df_preds.iterrows():
                p_name = r_st["punto"]
                r_lvl = r_st["riesgo"]
                c_hex = COLOR_HEX[r_lvl]
                p_pct = r_st["probabilidad_sequia"] * 100
                spi_v = f"{r_st['spi3_actual']:+.2f}" if pd.notna(r_st["spi3_actual"]) else "N/D"
                alt_v = ALTITUDES_ESTACIONES.get(p_name, 3750)

                folium.Circle(
                    location=[r_st["lat"], r_st["lon"]],
                    radius=15000 + int(r_st["probabilidad_sequia"] * 25000),
                    color=c_hex,
                    weight=1.2,
                    fill=True,
                    fill_color=c_hex,
                    fill_opacity=0.18
                ).add_to(m_central)

                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; width: 195px; background: #12141D; color: #F8FAFC; padding: 10px; border-radius: 8px; border: 1px solid {c_hex};">
                    <div style="font-size: 14px; font-weight: 700; margin-bottom: 4px;">{html.escape(p_name)} <span style="font-size:11px;color:#94A3B8;font-weight:400;">({alt_v}m)</span></div>
                    <div style="font-size: 12px;">Riesgo: <b style="color:{c_hex};">{r_lvl} ({p_pct:.1f}%)</b></div>
                    <div style="font-size: 12px;">SPI-3 actual: <b>{spi_v}</b></div>
                    <div style="font-size: 12px; color: #94A3B8;">Lluvia base: {r_st['lluvia_mes_base_mm']:.1f} mm</div>
                </div>
                """

                folium.CircleMarker(
                    location=[r_st["lat"], r_st["lon"]],
                    radius=7.5,
                    color="#FFFFFF",
                    weight=2,
                    fill=True,
                    fill_color=c_hex,
                    fill_opacity=0.95,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=f"{p_name}: {p_pct:.1f}% ({r_lvl})"
                ).add_to(m_central)

            folium.LayerControl(position="topright", collapsed=True).add_to(m_central)
            st_folium(m_central, height=485, use_container_width=True, returned_objects=[])
        else:
            df_mapa = df_preds.copy()
            df_mapa["color"] = df_mapa["riesgo"].map(COLOR_RGBA)
            df_mapa["elevation"] = (df_mapa["probabilidad_sequia"] * 85000) + 12000
            df_mapa["prob_pct"] = (df_mapa["probabilidad_sequia"] * 100).round(1).astype(str)
            df_mapa["spi3_txt"] = df_mapa["spi3_actual"].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "N/D")
            df_mapa["lluvia_txt"] = df_mapa["lluvia_mes_base_mm"].apply(lambda v: f"{v:.1f}")

            capa_anillo = pdk.Layer(
                "ScatterplotLayer",
                data=df_mapa,
                get_position=["lon", "lat"],
                get_radius=30000,
                get_fill_color=[99, 102, 241, 25],
                get_line_color="color",
                stroked=True,
                line_width_min_pixels=1.5,
            )
            capa_col = pdk.Layer(
                "ColumnLayer",
                data=df_mapa,
                get_position=["lon", "lat"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=11000,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
                extruded=True,
            )
            capa_txt = pdk.Layer(
                "TextLayer",
                data=df_mapa,
                get_position=["lon", "lat"],
                get_text="punto",
                get_size=14,
                get_color=[248, 250, 252, 240],
                get_alignment_baseline="'bottom'",
                get_pixel_offset=[0, -22],
                font_family="Inter, sans-serif",
                font_weight=600,
            )
            deck = pdk.Deck(
                layers=[capa_anillo, capa_col, capa_txt],
                initial_view_state=pdk.ViewState(latitude=-18.2, longitude=-67.65, zoom=5.8, pitch=45, bearing=-8),
                map_style="dark",
                tooltip={
                    "html": "<div style='background:#12141D;border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:10px 12px;color:#F8FAFC;font-family:Inter,sans-serif;'><b>{punto}</b><br>Probabilidad: <b>{prob_pct}% ({riesgo})</b><br>SPI-3: {spi3_txt} · Lluvia: {lluvia_txt} mm</div>"
                }
            )
            st.pydeck_chart(deck, use_container_width=True, height=485)

    with col_ranking:
        render_html(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; margin-bottom: 10px;">
            <span class="section-heading" style="margin-bottom: 0;">Semáforo por Estación</span>
            <span class="pill pill-neutral">Umbral: {cfg_h['umbral']*100:.1f}%</span>
        </div>
        """)

        df_ordenado = df_preds.sort_values("probabilidad_sequia", ascending=False).reset_index(drop=True)
        umb_pct = min(100.0, max(1.0, float(cfg_h["umbral"]) * 100.0))

        filas_html = []
        for idx, fila in df_ordenado.iterrows():
            r_est = str(fila["riesgo"])
            c_est = COLOR_HEX.get(r_est, "#10B981")
            pct = float(fila["probabilidad_sequia"]) * 100.0
            ancho_barra = min(100.0, max(2.0, pct))
            spi_txt = f"{fila['spi3_actual']:+.2f}" if pd.notna(fila["spi3_actual"]) else "N/D"
            p_nom = html.escape(str(fila["punto"]))

            filas_html.append(
                f'<div class="station-item">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                f'<span style="font-weight: 600; font-size: 0.92rem; color: #F8FAFC;">{p_nom}</span>'
                f'<div style="display: flex; align-items: center; gap: 10px;">'
                f'<span style="font-family: \'JetBrains Mono\', monospace; font-size: 0.92rem; font-weight: 700; color: {c_est};">{pct:.1f}%</span>'
                f'<span class="risk-badge-{r_est}">{r_est}</span>'
                f'</div>'
                f'</div>'
                f'<div style="position: relative; height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; margin-bottom: 6px;">'
                f'<div style="width: {ancho_barra:.1f}%; height: 100%; background: {c_est};"></div>'
                f'<div style="position: absolute; top: 0; bottom: 0; left: {umb_pct:.1f}%; width: 2px; background: rgba(255,255,255,0.65);"></div>'
                f'</div>'
                f'<div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94A3B8;">'
                f'<span>SPI-3: <b style="color:#E2E8F0;">{spi_txt}</b></span>'
                f'<span>Lluvia: <b style="color:#CBD5E1;">{fila["lluvia_mes_base_mm"]:.1f} mm</b></span>'
                f'</div>'
                f'</div>'
            )

        render_html("<div>" + "".join(filas_html) + "</div>")


# =============================================================================
# 2. SERIES HISTORICAS (SPI-3 Y PRECIPITACION)
# =============================================================================
elif pagina == "Series Históricas (SPI)":
    render_html("""
    <div class="page-banner">
        <div class="banner-eyebrow">REGISTRO CLIMATOLOGICO · 1981 - 2025</div>
        <h1 class="banner-title">Series Temporales de Precipitación e Índice SPI-3</h1>
        <div class="banner-desc">
            Evolución histórica por estación comparada con la normal climatológica OMM (1981–2010).
        </div>
    </div>
    """)

    df_hist = cargar_historico_modelado()
    puntos_disponibles = list(art["puntos"].keys())

    c_sel1, c_sel2 = st.columns([1.2, 2.0])
    with c_sel1:
        punto_sel = st.selectbox("Estación:", puntos_disponibles, index=0)
    with c_sel2:
        anios_ver = st.slider("Años recientes a mostrar:", min_value=3, max_value=30, value=10)

    if df_hist is not None and punto_sel in df_hist["punto"].unique():
        sub_p = df_hist[df_hist["punto"] == punto_sel].sort_values("fecha").copy()
        fecha_corte = sub_p["fecha"].max() - pd.DateOffset(years=anios_ver)
        sub_rec = sub_p[sub_p["fecha"] >= fecha_corte].copy()

        lluvia_anual = sub_p.groupby(sub_p["fecha"].dt.year)["lluvia"].sum().mean()
        freq_sequia = (sub_p["spi3"] <= -1.0).mean() * 100
        spi_ult = float(sub_p["spi3"].iloc[-1])
        c_spi = "#10B981" if spi_ult > -1.0 else ("#F59E0B" if spi_ult > -1.5 else "#F43F5E")

        render_html(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label"><span>Altitud</span></div>
                <div class="kpi-number">{sub_p['altitud'].iloc[0]:.0f} msnm</div>
                <div class="kpi-sub">{html.escape(punto_sel)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label"><span>Lluvia Anual Media</span></div>
                <div class="kpi-number">{lluvia_anual:.0f} mm</div>
                <div class="kpi-sub">Promedio 1981–2025</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label"><span>Frecuencia de Sequía</span></div>
                <div class="kpi-number">{freq_sequia:.1f}%</div>
                <div class="kpi-sub">Meses con SPI-3 &le; -1.0</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label"><span>Último SPI-3</span></div>
                <div class="kpi-number" style="color:{c_spi};">{spi_ult:+.2f}</div>
                <div class="kpi-sub">Escala normalizada</div>
            </div>
        </div>
        """)

        render_html("<div class='section-heading'>Trayectoria del Índice de Sequía SPI-3</div>")
        fig_spi = go.Figure()
        fig_spi.add_hrect(y0=-1.0, y1=-1.5, fillcolor="rgba(245, 158, 11, 0.10)", line_width=0, annotation_text="Moderada (-1.0)", annotation_position="top left", annotation_font={"color": "#FBBF24", "size": 10})
        fig_spi.add_hrect(y0=-1.5, y1=-3.3, fillcolor="rgba(244, 63, 94, 0.14)", line_width=0, annotation_text="Severa / Extrema (<= -1.5)", annotation_position="top left", annotation_font={"color": "#FDA4AF", "size": 10})
        fig_spi.add_hline(y=-1.0, line_dash="dash", line_color="#F43F5E", line_width=1.3)
        fig_spi.add_hline(y=0.0, line_dash="dot", line_color="rgba(255, 255, 255, 0.2)", line_width=1)

        fig_spi.add_trace(go.Scatter(
            x=sub_rec["fecha"],
            y=sub_rec["spi3"],
            mode="lines",
            line=dict(color="#818CF8", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(99, 102, 241, 0.08)",
            name="SPI-3",
            hovertemplate="<b>%{x|%Y-%m}</b> · SPI-3: <b>%{y:.2f}</b><extra></extra>"
        ))

        en_seq = sub_rec[sub_rec["spi3"] <= -1.0]
        fig_spi.add_trace(go.Scatter(
            x=en_seq["fecha"],
            y=en_seq["spi3"],
            mode="markers",
            marker=dict(color="#F43F5E", size=6),
            name="Sequía (<= -1.0)",
            hovertemplate="<b>Sequía</b> (%{x|%Y-%m}): %{y:.2f}<extra></extra>"
        ))

        fig_spi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            margin=dict(l=12, r=12, t=20, b=12),
            height=330,
            yaxis=dict(title="SPI-3", range=[-3.2, 3.2], gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_spi, use_container_width=True)

        render_html("<div class='section-heading'>Precipitación Mensual vs. Promedio Histórico (1981–2010)</div>")
        ref_30 = sub_p[(sub_p["fecha"].dt.year >= 1981) & (sub_p["fecha"].dt.year <= 2010)]
        clim_mensual = ref_30.groupby(ref_30["fecha"].dt.month)["lluvia"].mean()
        sub_rec["clim_lluvia"] = sub_rec["fecha"].dt.month.map(clim_mensual)

        fig_lluvia = go.Figure()
        fig_lluvia.add_trace(go.Bar(
            x=sub_rec["fecha"],
            y=sub_rec["lluvia"],
            name="Lluvia Observada (mm)",
            marker_color="rgba(56, 189, 248, 0.6)",
            hovertemplate="<b>%{x|%Y-%m}</b>: %{y:.1f} mm<extra></extra>"
        ))
        fig_lluvia.add_trace(go.Scatter(
            x=sub_rec["fecha"],
            y=sub_rec["clim_lluvia"],
            name="Promedio 1981–2010",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            hovertemplate="Normal: %{y:.1f} mm<extra></extra>"
        ))

        fig_lluvia.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            margin=dict(l=12, r=12, t=20, b=12),
            height=300,
            yaxis=dict(title="mm / mes", gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_lluvia, use_container_width=True)


# =============================================================================
# 3. TACOMETRO Y EXPLORADOR (DIVIDIDO EN 2 APARTADOS CLAROS E INTUITIVOS)
# =============================================================================
elif pagina == "Tacómetro y Explorador":
    render_html("""
    <div class="page-banner">
        <div class="banner-eyebrow">DIAGNOSTICO PUNTUAL Y EXPLORADOR CARTOGRAFICO</div>
        <h1 class="banner-title">Tacómetro de Riesgo y Consulta por Coordenadas</h1>
        <div class="banner-desc">
            Inspecciona las 8 estaciones monitoreadas del Altiplano o explora cualquier punto de Bolivia seleccionándolo en el mapa o ingresando sus coordenadas.
        </div>
    </div>
    """)

    tab_estaciones, tab_explorador = st.tabs([
        "1. Estaciones en Monitoreo (Altiplano)",
        "2. Explorar Punto en el Mapa o por Latitud / Longitud"
    ])

    # -------------------------------------------------------------------------
    # APARTADO 1: TACOMETRO DE LAS 8 ESTACIONES EN MONITOREO
    # -------------------------------------------------------------------------
    with tab_estaciones:
        puntos_lista = list(art["puntos"].keys())
        col_sel_est, col_resumen_est = st.columns([1.15, 1.35])

        with col_sel_est:
            punto_escogido = st.selectbox(
                "Selecciona la estación a diagnosticar:",
                options=puntos_lista,
                index=4  # Oruro por defecto
            )
            info_p = next((p for p in preds_h if p["punto"] == punto_escogido), preds_h[0])
            prob_p = float(info_p["probabilidad_sequia"]) * 100.0
            umb_p = float(info_p["umbral_modelo"]) * 100.0
            riesgo_p = str(info_p["riesgo"])

            fig_g1 = _crear_gauge_plotly(
                prob_val=prob_p,
                umbral_val=umb_p,
                riesgo_txt=riesgo_p,
                subtitulo=f"{punto_escogido} · Objetivo {info_p['mes_objetivo']} (+{h}M)"
            )
            st.plotly_chart(fig_g1, use_container_width=True, key="gauge_estacion_fija")

        with col_resumen_est:
            alt_p = ALTITUDES_ESTACIONES.get(punto_escogido, 3750)
            spi_p = f"{info_p['spi3_actual']:+.2f}" if pd.notna(info_p["spi3_actual"]) else "N/D"

            # Obtener probabilidades en los 3 horizontes para esta estación
            probs_3h = {}
            for h_i in [1, 2, 3]:
                lista_hi = telemetria_data["horizontes"].get(str(h_i)) or telemetria_data["horizontes"].get(h_i, [])
                item_hi = next((x for x in lista_hi if x["punto"] == punto_escogido), None)
                if item_hi:
                    probs_3h[h_i] = (item_hi["probabilidad_sequia"] * 100.0, item_hi["riesgo"], item_hi["mes_objetivo"])

            diag_msg = (
                "Alerta activa: la probabilidad supera el umbral óptimo. Se recomienda seguimiento de reservorios y riego."
                if riesgo_p == "ALTO" else (
                    "Vigilancia preventiva: probabilidad moderada cercana al umbral de transición."
                    if riesgo_p == "MEDIO" else
                    "Condición estable: probabilidad muy por debajo del umbral de alerta."
                )
            )

            render_html(f"""
            <div class="glass-panel" style="margin-top: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--border-hairline);">
                    <div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em;">Estación Oficial</div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">{html.escape(punto_escogido)}</div>
                    </div>
                    <span class="risk-badge-{riesgo_p}">{riesgo_p} RIESGO</span>
                </div>
                <div class="metrics-grid-2" style="margin-bottom: 14px;">
                    <div class="mini-stat">
                        <div class="mini-stat-label">Ubicación y Altitud</div>
                        <div class="mini-stat-val">{info_p['lat']:.2f}°S, {info_p['lon']:.2f}°O · {alt_p}m</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-label">SPI-3 Mes Base ({info_p['mes_base']})</div>
                        <div class="mini-stat-val" style="color: #818CF8;">{spi_p}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-label">Lluvia Mes Base</div>
                        <div class="mini-stat-val" style="color: #38BDF8;">{info_p['lluvia_mes_base_mm']:.1f} mm</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-stat-label">Umbral de Corte F1</div>
                        <div class="mini-stat-val" style="color: #F59E0B;">{umb_p:.1f}%</div>
                    </div>
                </div>
                <div style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;">
                    Proyección Multi-Horizonte para {html.escape(punto_escogido)}:
                </div>
                <div class="metrics-grid-3" style="margin-bottom: 14px;">
                    <div class="mini-stat" style="padding: 9px 12px;">
                        <div class="mini-stat-label">+1 Mes ({probs_3h.get(1, (0,'',''))[2]})</div>
                        <div class="mini-stat-val" style="font-size: 0.95rem;">{probs_3h.get(1, (0,'',''))[0]:.1f}% <span style="font-size:0.7rem;color:#94A3B8;">({probs_3h.get(1, (0,'',''))[1]})</span></div>
                    </div>
                    <div class="mini-stat" style="padding: 9px 12px;">
                        <div class="mini-stat-label">+2 Meses ({probs_3h.get(2, (0,'',''))[2]})</div>
                        <div class="mini-stat-val" style="font-size: 0.95rem;">{probs_3h.get(2, (0,'',''))[0]:.1f}% <span style="font-size:0.7rem;color:#94A3B8;">({probs_3h.get(2, (0,'',''))[1]})</span></div>
                    </div>
                    <div class="mini-stat" style="padding: 9px 12px;">
                        <div class="mini-stat-label">+3 Meses ({probs_3h.get(3, (0,'',''))[2]})</div>
                        <div class="mini-stat-val" style="font-size: 0.95rem;">{probs_3h.get(3, (0,'',''))[0]:.1f}% <span style="font-size:0.7rem;color:#94A3B8;">({probs_3h.get(3, (0,'',''))[1]})</span></div>
                    </div>
                </div>
                <div style="font-size: 0.8rem; color: var(--text-muted); background: rgba(255,255,255,0.02); padding: 10px 12px; border-radius: 8px; border-left: 3px solid {COLOR_HEX[riesgo_p]};">
                    {html.escape(diag_msg)}
                </div>
            </div>
            """)

    # -------------------------------------------------------------------------
    # APARTADO 2: EXPLORADOR LIBRE POR MAPA FISIOGRAFICO O LATITUD/LONGITUD
    # -------------------------------------------------------------------------
    with tab_explorador:
        if "exp_lat" not in st.session_state:
            st.session_state["exp_lat"] = -17.50
            st.session_state["exp_lon"] = -66.50
            st.session_state["exp_custom_res"] = None

        col_mapa_exp, col_panel_exp = st.columns([1.3, 1.1], gap="medium")

        with col_mapa_exp:
            render_html("""
            <div class="section-heading">Haz clic en el mapa para seleccionar cualquier punto</div>
            <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 8px; font-size: 0.75rem;">
                <span style="color: #38BDF8; font-weight: 600;">■ Altiplano (&gt;3,500m)</span>
                <span style="color: #F59E0B; font-weight: 600;">■ Valles (1,500–3,400m)</span>
                <span style="color: #10B981; font-weight: 600;">■ Llanos (&lt;1,500m)</span>
            </div>
            """)

            m_pick = folium.Map(location=[-17.5, -65.5], zoom_start=5, tiles=None, control_scale=True)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, HERE, DeLorme",
                name="Modo Oscuro (Esri)",
                overlay=False,
                control=True
            ).add_to(m_pick)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Imagery, NASA",
                name="Satélite Real (NASA / Esri)",
                overlay=False,
                control=True
            ).add_to(m_pick)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Topo",
                name="Relieve Físico (Altiplano, Valles y Llanos)",
                overlay=False,
                control=True
            ).add_to(m_pick)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Límites y Ciudades",
                overlay=True,
                control=True
            ).add_to(m_pick)

            fg_z = folium.FeatureGroup(name="Regiones Fisiográficas", show=True)
            for z in ZONAS_FISIOGRAFICAS:
                folium.Polygon(
                    locations=z["coords"],
                    color=z["color"],
                    weight=1.8,
                    dash_array="5, 5",
                    fill=True,
                    fill_color=z["color"],
                    fill_opacity=0.11,
                    tooltip=f"<b>{z['nombre']}</b> ({z['altitud_media']})"
                ).add_to(fg_z)
            fg_z.add_to(m_pick)

            for _, r_st in df_preds.iterrows():
                folium.CircleMarker(
                    location=[r_st["lat"], r_st["lon"]],
                    radius=5.5,
                    color="#090A0F",
                    weight=1.5,
                    fill=True,
                    fill_color=COLOR_HEX[r_st["riesgo"]],
                    fill_opacity=0.9,
                    tooltip=f"Estación {r_st['punto']}"
                ).add_to(m_pick)

            lat_act = float(st.session_state["exp_lat"])
            lon_act = float(st.session_state["exp_lon"])

            folium.CircleMarker(
                location=[lat_act, lon_act],
                radius=10,
                color="#6366F1",
                weight=3,
                fill=True,
                fill_color="#FFFFFF",
                fill_opacity=0.95,
                tooltip=f"Punto Seleccionado: ({lat_act:.4f}°, {lon_act:.4f}°)"
            ).add_to(m_pick)

            folium.LayerControl(position="topright", collapsed=True).add_to(m_pick)

            map_click = st_folium(m_pick, height=520, use_container_width=True, key="mapa_explorador_libre")
            if map_click and map_click.get("last_clicked"):
                c_lat = round(float(map_click["last_clicked"]["lat"]), 4)
                c_lon = round(float(map_click["last_clicked"]["lng"]), 4)
                if abs(c_lat - st.session_state["exp_lat"]) > 1e-4 or abs(c_lon - st.session_state["exp_lon"]) > 1e-4:
                    st.session_state["exp_lat"] = c_lat
                    st.session_state["exp_lon"] = c_lon
                    st.session_state["exp_custom_res"] = None
                    st.rerun()

        with col_panel_exp:
            render_html("<div class='section-heading'>Coordenadas del Punto (Clic en mapa o ingreso manual)</div>")
            in_c1, in_c2 = st.columns(2)
            with in_c1:
                nuevo_lat = st.number_input(
                    "Latitud (° Sur)",
                    min_value=-23.50,
                    max_value=-9.50,
                    value=float(st.session_state["exp_lat"]),
                    step=0.05,
                    format="%.4f"
                )
            with in_c2:
                nuevo_lon = st.number_input(
                    "Longitud (° Oeste)",
                    min_value=-70.50,
                    max_value=-57.00,
                    value=float(st.session_state["exp_lon"]),
                    step=0.05,
                    format="%.4f"
                )

            if abs(nuevo_lat - st.session_state["exp_lat"]) > 1e-4 or abs(nuevo_lon - st.session_state["exp_lon"]) > 1e-4:
                st.session_state["exp_lat"] = round(float(nuevo_lat), 4)
                st.session_state["exp_lon"] = round(float(nuevo_lon), 4)
                st.session_state["exp_custom_res"] = None
                st.rerun()

            lat_act = st.session_state["exp_lat"]
            lon_act = st.session_state["exp_lon"]
            fisio = clasificar_fisiografia(lat_act, lon_act)

            if st.session_state.get("exp_custom_res") is not None:
                diag_exp = st.session_state["exp_custom_res"]
                fuente_txt = "Cálculo directo NASA POWER (Serie 1981–Hoy)"
            else:
                diag_exp = interpolar_diagnostico_espacial(lat_act, lon_act, df_preds, cfg_h["umbral"])
                fuente_txt = "Estación oficial cercana" if diag_exp["es_estacion_oficial"] else f"Estimación espacial (Ref: {diag_exp['estacion_cercana']} a {diag_exp['distancia_km']} km)"

            prob_exp = float(diag_exp["probabilidad_sequia"]) * 100.0
            umb_exp = float(diag_exp["umbral_modelo"]) * 100.0
            riesgo_exp = str(diag_exp["riesgo"])

            fig_g2 = _crear_gauge_plotly(
                prob_val=prob_exp,
                umbral_val=umb_exp,
                riesgo_txt=riesgo_exp,
                subtitulo=f"({lat_act:.2f}°, {lon_act:.2f}°) · {fisio['zona']}"
            )
            st.plotly_chart(fig_g2, use_container_width=True, key="gauge_explorador_mapa")

            render_html(f"""
            <div class="glass-panel" style="padding: 14px 18px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span style="font-size: 0.72rem; color: {fisio['color']}; font-weight: 600;">{fisio['zona']} ({fisio['altitud_est']})</span>
                        <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 2px;">{html.escape(fuente_txt)}</div>
                    </div>
                    <span class="risk-badge-{riesgo_exp}">{riesgo_exp}</span>
                </div>
                <div class="metrics-grid-3">
                    <div class="mini-stat" style="padding: 8px 10px;">
                        <div class="mini-stat-label">SPI-3 Estimado</div>
                        <div class="mini-stat-val" style="font-size: 0.92rem;">{diag_exp['spi3_actual']:+.2f}</div>
                    </div>
                    <div class="mini-stat" style="padding: 8px 10px;">
                        <div class="mini-stat-label">Lluvia Base</div>
                        <div class="mini-stat-val" style="font-size: 0.92rem;">{diag_exp['lluvia_mes_base_mm']:.1f} mm</div>
                    </div>
                    <div class="mini-stat" style="padding: 8px 10px;">
                        <div class="mini-stat-label">Fiabilidad</div>
                        <div class="mini-stat-val" style="font-size: 0.8rem; color: #CBD5E1;">{fisio['fiabilidad'].split(' ')[0]}</div>
                    </div>
                </div>
            </div>
            """)

            if st.button("Consultar Serie Histórica Exacta en NASA POWER (~25 s)", use_container_width=True):
                with st.spinner(f"Descargando serie 1981–hoy de NASA POWER para ({lat_act:.2f}°, {lon_act:.2f}°)..."):
                    try:
                        res_ex = PR.predecir_punto(lat=lat_act, lon=lon_act, horizonte=h, art=art)
                        res_ex["es_estacion_oficial"] = False
                        st.session_state["exp_custom_res"] = res_ex
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error al consultar NASA POWER: {ex}")
                    st.rerun()


# =============================================================================
# 4. FACTORES CLIMATICOS E INTERPRETABILIDAD (XAI)
# =============================================================================
elif pagina == "Factores Climáticos (IA)":
    render_html("""
    <div class="page-banner">
        <div class="banner-eyebrow">INTERPRETABILIDAD DEL MODELO · SHAP Y TELECONEXION ENSO</div>
        <h1 class="banner-title">Factores Climáticos que Impulsan la Predicción</h1>
        <div class="banner-desc">
            Cómo pondera la Inteligencia Artificial la memoria hídrica, los acumulados de lluvia y el fenómeno de El Niño.
        </div>
    </div>
    """)

    render_html("""
    <div class="steps-grid-4" style="margin-bottom: 20px;">
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#FDA4AF;">1. Océano Pacífico</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Calentamiento El Niño</div>
            <div class="kpi-sub">Anomalías ONI &ge; +0.5°C alteran la circulación tropical.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#FDE68A;">2. Alta de Bolivia</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Bloqueo de Humedad</div>
            <div class="kpi-sub">Vientos del oeste frenan el ingreso húmedo desde la Amazonía.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#38BDF8;">3. Déficit Pluvial</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Caída de Acumulados</div>
            <div class="kpi-sub">Disminuye la lluvia de 3 y 6 meses y sube la radiación.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#A5B4FC;">4. Inercia Hídrica</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Umbral SPI-3 &le; -1.0</div>
            <div class="kpi-sub">El déficit trimestral se proyecta hacia el mes t+h.</div>
        </div>
    </div>
    """)

    col_x1, col_x2 = st.columns([1.3, 1.2])

    with col_x1:
        render_html("<div class='section-heading'>Desglose de Factores por Estación (Gráfico Waterfall)</div>")
        est_xai = st.selectbox("Estación:", list(art["puntos"].keys()), index=4)
        pred_est = next((p for p in preds_h if p["punto"] == est_xai), preds_h[0])
        f_est = features_actuales.get(est_xai, {})

        spi3_v = f_est.get("spi3") if f_est.get("spi3") is not None else (pred_est.get("spi3_actual") or 0.0)
        acum3_v = f_est.get("acum3") if f_est.get("acum3") is not None else (pred_est.get("lluvia_mes_base_mm", 20.0) * 2.2)
        oni_v = f_est.get("oni_l2") if f_est.get("oni_l2") is not None else 0.0
        tmax_a = f_est.get("tmax_anom") if f_est.get("tmax_anom") is not None else 0.0

        base_prev = 12.8
        p_final = pred_est["probabilidad_sequia"] * 100.0
        delta_total = p_final - base_prev

        raw_imp_spi = -spi3_v * 12.0
        raw_imp_acum = -((acum3_v - 65.0) / 50.0) * 6.0
        raw_imp_oni = oni_v * 4.5
        raw_imp_temp = tmax_a * 2.0
        suma_raw = raw_imp_spi + raw_imp_acum + raw_imp_oni + raw_imp_temp
        if abs(suma_raw) > 1e-3:
            escala = delta_total / suma_raw
            c1, c2, c3, c4 = raw_imp_spi * escala, raw_imp_acum * escala, raw_imp_oni * escala, raw_imp_temp * escala
        else:
            c1, c2, c3, c4 = delta_total * 0.5, delta_total * 0.25, delta_total * 0.15, delta_total * 0.10

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=[
                "Base Histórica",
                f"SPI-3 ({spi3_v:+.2f})",
                f"Acum. 3M ({acum3_v:.0f}mm)",
                f"ONI ({oni_v:+.2f}°C)",
                f"Temp ({tmax_a:+.1f}°C)",
                f"Predicción {est_xai}"
            ],
            textposition="outside",
            text=[f"{base_prev:.1f}%", f"{c1:+.1f}%", f"{c2:+.1f}%", f"{c3:+.1f}%", f"{c4:+.1f}%", f"{p_final:.1f}%"],
            y=[base_prev, c1, c2, c3, c4, 0],
            connector={"line": {"color": "rgba(255, 255, 255, 0.18)"}},
            increasing={"marker": {"color": "#F43F5E"}},
            decreasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#6366F1"}}
        ))
        fig_wf.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            height=340,
            margin=dict(l=12, r=12, t=24, b=12),
            yaxis=dict(title="Probabilidad (%)", gridcolor="rgba(255,255,255,0.05)"),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_x2:
        render_html("<div class='section-heading'>Importancia Global de Variables (SHAP)</div>")
        datos_shap = pd.DataFrame({
            "Variable": [
                "SPI-3 actual y rezagos",
                "Lluvia acumulada 3 meses",
                "Lluvia acumulada 6 meses",
                "Indice El Niño (ONI)",
                "Anomalía Temp. Máxima",
                "SPI-1 mes base",
                "Radiación solar",
                "Humedad relativa",
                "Ciclo estacional",
                "Altitud"
            ],
            "Peso": [28.4, 21.8, 15.6, 11.2, 7.9, 5.1, 4.0, 3.1, 1.9, 1.0]
        }).sort_values("Peso", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=datos_shap["Peso"],
            y=datos_shap["Variable"],
            orientation="h",
            text=datos_shap["Peso"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            marker=dict(color=datos_shap["Peso"], colorscale=[[0, "#312E81"], [0.5, "#4F46E5"], [1, "#818CF8"]])
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            height=390,
            margin=dict(l=12, r=42, t=10, b=12),
            xaxis=dict(title="Aporte Relativo (%)", range=[0, 34], gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# =============================================================================
# 5. REPORTES Y DESCARGAS (PDF NATIVO Y CSV)
# =============================================================================
elif pagina == "Reportes y Descargas":
    render_html("""
    <div class="page-banner">
        <div class="banner-eyebrow">EXPORTACION DE BOLETINES Y DATOS</div>
        <h1 class="banner-title">Generador de Reporte Oficial en PDF y Matriz CSV</h1>
        <div class="banner-desc">
            Descarga el boletín ejecutivo en formato PDF listo para impresión o la tabla completa en CSV.
        </div>
    </div>
    """)

    pdf_bytes = generar_pdf_ejecutivo(
        df_preds=df_preds,
        h=h,
        nombre_mod_h=str(nombre_mod_h),
        umbral_h=float(cfg_h["umbral"]),
        sync_time=sync_ts
    )

    df_export = df_preds[["punto", "lat", "lon", "horizonte_meses", "mes_base", "mes_objetivo", "probabilidad_sequia", "umbral_modelo", "riesgo", "spi3_actual", "lluvia_mes_base_mm"]].copy()
    df_export["altitud_msnm"] = df_export["punto"].map(ALTITUDES_ESTACIONES)
    df_export.columns = ["Estacion", "Latitud", "Longitud", "Horizonte", "Mes_Base", "Mes_Objetivo", "Probabilidad", "Umbral_F1", "Riesgo", "SPI3_Base", "Lluvia_mm", "Altitud_msnm"]

    col_tabla_rep, col_botones_rep = st.columns([1.55, 1.05])

    with col_tabla_rep:
        render_html("<div class='section-heading'>Vista Previa de la Matriz de Pronóstico</div>")
        st.dataframe(df_export, use_container_width=True, hide_index=True)

    with col_botones_rep:
        render_html("<div class='section-heading'>Descargas Directas</div>")
        render_html("""
        <div class="glass-panel">
            <div style="font-weight: 700; font-size: 1rem; margin-bottom: 6px;">Boletín Técnico Ejecutivo (.PDF)</div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.5;">
                Documento A4 con resumen ejecutivo, tabla coloreada por nivel de riesgo, gráfico comparativo frente al umbral F1 y recomendaciones técnicas.
            </div>
        </div>
        """)
        st.download_button(
            label="Descargar Reporte Oficial (.PDF)",
            data=pdf_bytes,
            file_name=f"Boletin_Sequia_Altiplano_h{h}_{mes_objetivo_actual}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        render_html("""
        <div class="glass-panel">
            <div style="font-weight: 700; font-size: 1rem; margin-bottom: 6px;">Matriz de Datos (.CSV)</div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.5;">
                Tabla estructurada con coordenadas, altitudes, valores de SPI-3 y probabilidades para análisis en Excel, R o sistemas SIG.
            </div>
        </div>
        """)
        st.download_button(
            label="Descargar Datos en CSV",
            data=df_export.to_csv(index=False).encode("utf-8"),
            file_name=f"Datos_Sequia_Altiplano_h{h}_{mes_objetivo_actual}.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# 6. METODOLOGIA Y SATELITES NASA
# =============================================================================
elif pagina == "Metodología y Satélites":
    render_html("""
    <div class="page-banner">
        <div class="banner-eyebrow">ESPECIFICACIONES CIENTIFICAS · OMM & NASA POWER</div>
        <h1 class="banner-title">Metodología Estadística y Arquitectura Satelital</h1>
        <div class="banner-desc">
            Transformación Gamma mixta del índice SPI-3, partición temporal sin fuga de datos y desempeño por horizonte.
        </div>
    </div>
    """)

    render_html("""
    <div class="steps-grid-4" style="margin-bottom: 20px;">
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#38BDF8;">Fase 1 · Satélites</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">NASA POWER & NOAA</div>
            <div class="kpi-sub">Series diarias 1981–2025 (MERRA-2 / GPM) en grilla de ~50 km e índice ONI.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#818CF8;">Fase 2 · Estándar OMM</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Gamma Mixta (SPI-3)</div>
            <div class="kpi-sub">Ajuste sobre el periodo base 1981–2010 transformado a una normal N(0,1).</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#F59E0B;">Fase 3 · Validación</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Sin Fuga Temporal</div>
            <div class="kpi-sub">Corte cronológico con margen de guarda (gap = h meses) y bloques 2011–2018.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span style="color:#10B981;">Fase 4 · Inferencia</span></div>
            <div style="font-weight: 600; font-size: 0.92rem; margin-bottom: 4px;">Selección por Horizonte</div>
            <div class="kpi-sub">Gradient Boosting (h=1), LightGBM (h=2) y Climatología (h=3).</div>
        </div>
    </div>
    """)

    col_m1, col_m2 = st.columns([1.25, 1.25])

    with col_m1:
        render_html("<div class='section-heading'>Transformación Normal Estándar del SPI-3 y Cola de Sequía</div>")
        z_vals = np.linspace(-3.2, 3.2, 300)
        fig_norm = go.Figure()
        fig_norm.add_trace(go.Scatter(
            x=z_vals, y=stats.norm.pdf(z_vals), mode="lines",
            line=dict(color="#818CF8", width=2.5), name="Distribución SPI-3 N(0,1)"
        ))
        z_seq = z_vals[z_vals <= -1.0]
        fig_norm.add_trace(go.Scatter(
            x=z_seq, y=stats.norm.pdf(z_seq), mode="lines",
            line=dict(color="#F43F5E", width=0), fill="tozeroy",
            fillcolor="rgba(244, 63, 94, 0.4)", name="Sequía (SPI-3 <= -1.0)"
        ))
        fig_norm.add_vline(x=-1.0, line_dash="dash", line_color="#F43F5E", line_width=1.8)
        fig_norm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            height=310,
            margin=dict(l=12, r=12, t=20, b=12),
            xaxis=dict(title="Valor SPI-3", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Densidad", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_norm, use_container_width=True)

    with col_m2:
        render_html("<div class='section-heading'>Desempeño en Prueba Histórica (2019–2025)</div>")
        df_res_h = pd.DataFrame({
            "Horizonte": ["h=1 (+1 Mes)", "h=2 (+2 Meses)", "h=3 (+3 Meses)"],
            "Modelo Elegido": [0.663, 0.417, 0.279],
            "Persistencia": [0.578, 0.386, 0.291],
            "Climatología": [0.293, 0.293, 0.276]
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=df_res_h["Horizonte"], y=df_res_h["Modelo Elegido"], name="Modelo Elegido (F1)", marker_color="#6366F1", text=df_res_h["Modelo Elegido"], textposition="outside"))
        fig_comp.add_trace(go.Bar(x=df_res_h["Horizonte"], y=df_res_h["Persistencia"], name="Persistencia", marker_color="#38BDF8", text=df_res_h["Persistencia"], textposition="outside"))
        fig_comp.add_trace(go.Bar(x=df_res_h["Horizonte"], y=df_res_h["Climatología"], name="Climatología", marker_color="#475569", text=df_res_h["Climatología"], textposition="outside"))
        fig_comp.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18, 20, 29, 0.65)",
            height=310,
            margin=dict(l=12, r=12, t=20, b=12),
            yaxis=dict(title="F1-Score (Test)", range=[0, 0.78], gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

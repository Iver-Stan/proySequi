"""Configuración central del proyecto (todo lo ajustable vive aquí)."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"
FIGURAS = RAIZ / "reports" / "figuras"
for _d in (DATA_RAW, DATA_PROC, MODELOS, FIGURAS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- puntos
# Separados >= ~0.7° entre sí: NASA POWER usa una grilla de ~0.5° x 0.625°,
# y dos puntos más cercanos (p. ej. La Paz y Viacha) caen en la MISMA celda
# y producirían series idénticas (datos duplicados).
PUNTOS = {
    "La Paz":      (-16.50, -68.15),
    "Copacabana":  (-16.17, -69.09),
    "Charaña":     (-17.59, -69.45),
    "Patacamaya":  (-17.24, -67.92),
    "Oruro":       (-17.97, -67.11),
    "Challapata":  (-18.90, -66.77),
    "Potosí":      (-19.58, -65.75),
    "Uyuni":       (-20.46, -66.82),
}

# ---------------------------------------------------------------- datos
FECHA_INI = "19810101"
FECHA_FIN = "20251231"
PARAMS_POWER = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN",
                "RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN"]
RENOMBRAR = {"PRECTOTCORR": "lluvia", "T2M": "t2m", "T2M_MAX": "tmax",
             "T2M_MIN": "tmin", "RH2M": "rh2m", "WS2M": "ws2m",
             "PS": "ps", "ALLSKY_SFC_SW_DWN": "rad"}
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ONI_URL_HTML = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/"
ONI_URLS_TXT = ["https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"]
ONI_ARCHIVO_MANUAL = DATA_RAW / "oni_manual.txt"   # plan B (ver README)

# ---------------------------------------------------------------- SPI / objetivo
REF_INI, REF_FIN = 1981, 2010        # periodo de referencia OMM (30 años)
UMBRAL_SEQUIA = -1.0                 # SPI-3 <= -1  -> sequía moderada o peor
HORIZONTES = [1, 2, 3]
ESCALAS_SPI = [1, 3, 6]
ONI_REZAGOS = [2, 3]                 # meses (el ONI se publica con retraso y se revisa)

# ---------------------------------------------------------------- particiones
TRAIN_FIN = "2012-12-31"
VAL_FIN = "2018-12-31"               # test = 2019 en adelante
SEMILLA = 42

FEATURES_ESTATICAS = ["altitud"]     # lat/lon se omiten: con 8 puntos identifican al punto (sobreajuste)

# Modo de prueba SIN internet (datos sintéticos, SOLO para verificar que el código corre)
MODO_SINTETICO = os.environ.get("SEQUIA_SINTETICO", "0") == "1"

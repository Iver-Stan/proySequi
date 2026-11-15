"""Entrena el modelo final SIN notebook y guarda models/artefacto_sequia.joblib.

Uso:
    python scripts/entrenar_y_guardar.py                 # selección por horizonte
    python scripts/entrenar_y_guardar.py --modelo "Random Forest"  # fuerza un modelo
    SEQUIA_SINTETICO=1 python scripts/entrenar_y_guardar.py   # prueba offline (datos NO reales)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, features as F, modelos as M, pipeline as P, predecir as PR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", default="auto", help="'auto' selecciona un modelo distinto por horizonte")
ap.add_argument("--tratamiento", default="ninguno", choices=["ninguno", "log1p"])
args = ap.parse_args()

ds = P.construir_dataset()
d, feats = ds["d"], F.lista_features()
finales = {}
for h in C.HORIZONTES:
    candidatos = P.candidatos_horizonte(d, h, feats)
    if args.modelo == "auto":
        tabla = P.comparar_candidatos_post_referencia(d, h, candidatos,
                                                     args.tratamiento, filtro_comun=feats)
        nombre = P.elegir_candidato(tabla, h)
        pipe, columnas = candidatos[nombre]
    else:
        tr, _, _ = F.particionar(d, h, feats)
        prevalencia = tr[f"y_h{h}"].mean()
        catalogo = M.catalogo_modelos(scale_pos_weight=(1 - prevalencia) / max(prevalencia, 1e-6))
        if args.modelo not in catalogo:
            raise ValueError(f"Modelo desconocido: {args.modelo}. Opciones: {list(catalogo)}")
        nombre, pipe, columnas = args.modelo, catalogo[args.modelo], feats
    finales[h] = P.entrenar_final(d, h, pipe, columnas, args.tratamiento)
    finales[h]["nombre"] = nombre
    m = finales[h]["metricas_test"]
    print(f"h={h}: {nombre} | test F1={m['F1']:.3f}  PR-AUC={m['PR-AUC']:.3f}  ROC-AUC={m['ROC-AUC']:.3f}")
ruta = PR.guardar_artefacto(P.armar_artefacto(ds, finales, "por horizonte" if args.modelo == "auto" else args.modelo))
print("Guardado:", ruta)

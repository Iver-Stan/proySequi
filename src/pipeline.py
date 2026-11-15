"""Funciones de alto nivel compartidas por el notebook, el script de entrenamiento y la app."""
import datetime as dt

import pandas as pd
from sklearn.base import clone

from . import config as C
from . import datos, spi as spi_mod, features as F, modelos as M


def construir_dataset(forzar_descarga=False, verbose=True):
    """Descarga/lee datos -> limpia -> mensual -> SPI -> variables. Devuelve un diccionario."""
    if verbose:
        print("Modo SINTÉTICO (solo prueba, datos NO reales)" if C.MODO_SINTETICO else "Datos reales NASA POWER + NOAA ONI")
    raw = datos.descargar_todos(forzar=forzar_descarga,
                                archivo="power_diario_sintetico.csv" if C.MODO_SINTETICO else "power_diario.csv")
    diario, reporte = datos.limpiar_diario(raw)
    mensual = datos.a_mensual(diario)
    oni = datos.descargar_oni(forzar=forzar_descarga)
    params = {k: spi_mod.ajustar_parametros(mensual, k) for k in C.ESCALAS_SPI}
    ms = spi_mod.agregar_spi(mensual, params)
    clim = F.calcular_climatologia(mensual)
    d = F.construir_features(ms, oni, clim)
    return dict(raw=raw, diario=diario, reporte=reporte, mensual=mensual, oni=oni,
                spi_params=params, climatologia=clim, ms=ms, d=d)


def candidatos_horizonte(d, h, features, modelo_h1=None):
    """Modelos compactos para comparar por horizonte, incluida una línea base."""
    tr, va, _ = F.particionar(d, h, features)
    prevalencia = pd.concat([tr, va])[f"y_h{h}"].mean()
    peso = (1 - prevalencia) / max(prevalencia, 1e-6)
    catalogo = M.catalogo_modelos(scale_pos_weight=peso)
    nombres = ["Regresión logística", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]
    opciones = {n: (catalogo[n], list(features)) for n in nombres if n in catalogo}
    if "LightGBM" in catalogo:
        regulado = clone(catalogo["LightGBM"]).set_params(
            clf__num_leaves=7, clf__min_child_samples=50, clf__reg_lambda=2.0)
        opciones["LightGBM regulado"] = (regulado, list(features))
    if h == 1 and modelo_h1 is not None:
        opciones["Gradient Boosting ajustado"] = (clone(modelo_h1), list(features))
    opciones["Climatología"] = (M.ClimatologiaClassifier(), ["punto", "mes"])
    return opciones


def elegir_candidato(tabla, h):
    """Prioriza F1; en h=1 desempata modelos cercanos por capacidad de ordenar riesgo."""
    if h == 1:
        tabla = tabla[tabla["F1"] >= tabla["F1"].max() - 0.02]
        return tabla.sort_values(["PR-AUC", "F1"], ascending=False).index[0]
    return tabla.sort_values(["F1", "PR-AUC"], ascending=False).index[0]


def comparar_candidatos_post_referencia(d, h, candidatos, tratamiento="ninguno", filtro_comun=None):
    """Compara modelos con predicciones fuera de muestra de 2011-2018.

    `candidatos` asigna nombre -> (estimador, columnas). Los pliegues comienzan
    después de 2010, cuando ya está disponible la referencia SPI/climatológica.
    """
    tabla = F.tratar_outliers(d, tratamiento)
    if filtro_comun:
        tabla = tabla.dropna(subset=filtro_comun)
    filas = []
    for nombre, (estimador, columnas) in candidatos.items():
        tr, va, _ = F.particionar(tabla, h, columnas)
        tv = pd.concat([tr, va]).sort_values(["fecha", "punto"]).reset_index(drop=True)
        y = tv[f"y_h{h}"]
        cv = F.splits_post_referencia(tv["fecha"], h)
        probs, cub = M.oof_probs(estimador, tv[columnas], y, tv["fecha"], cv=cv)
        umbral = M.mejor_umbral(y.to_numpy()[cub], probs[cub])
        met = M.metricas(y.to_numpy()[cub], probs[cub], umbral)
        filas.append({"Modelo": nombre, **met, "umbral": umbral,
                      "fraccion_alertas": float((probs[cub] >= umbral).mean()),
                      "prevalencia": float(y.to_numpy()[cub].mean()), "n_oof": int(cub.sum())})
    return pd.DataFrame(filas).set_index("Modelo").sort_values("F1", ascending=False)


def entrenar_final(d, h, pipeline, features, tratamiento="ninguno"):
    """Entrena sobre train+val, elige umbral OOF posterior a 2010 y mide test una vez."""
    dt_ = F.tratar_outliers(d, tratamiento)
    tr, va, te = F.particionar(dt_, h, features)
    tv = pd.concat([tr, va]).sort_values(["fecha", "punto"]).reset_index(drop=True)
    y = f"y_h{h}"
    cv = F.splits_post_referencia(tv["fecha"], h)
    pcv, cub = M.oof_probs(pipeline, tv[features], tv[y], tv["fecha"], cv=cv)
    umbral = M.mejor_umbral(tv[y].values[cub], pcv[cub])
    modelo = clone(pipeline).fit(tv[features], tv[y])
    pte = M.probs(modelo, te[features])
    return {"pipeline": modelo, "umbral": umbral, "features": list(features), "tratamiento": tratamiento,
            "metricas_test": M.metricas(te[y], pte, umbral), "p_test": pte, "y_test": te[y].values,
            "test_idx": te.index, "n_train": len(tv), "n_test": len(te)}


def armar_artefacto(ds, finales, nombre_modelo):
    """Empaqueta lo necesario para predecir en producción."""
    return {
        "version": f"v1-{dt.date.today():%Y%m%d}",
        "creado": dt.datetime.now().isoformat(timespec="seconds"),
        "sintetico": C.MODO_SINTETICO,
        "modelo": nombre_modelo,
        "modelos": {h: {k: v for k, v in f.items() if k in ("pipeline", "umbral", "features", "tratamiento", "metricas_test", "nombre")}
                    for h, f in finales.items()},
        "spi_params": ds["spi_params"],
        "climatologia": ds["climatologia"],
        "puntos": dict(C.PUNTOS),
        "config": {"umbral_sequia": C.UMBRAL_SEQUIA, "ref": (C.REF_INI, C.REF_FIN),
                   "train_fin": C.TRAIN_FIN, "val_fin": C.VAL_FIN},
    }

"""Catálogo de modelos, espacios de búsqueda y métricas de evaluación."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, precision_recall_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import splits_panel

try:
    from xgboost import XGBClassifier
except Exception:                                   # opcional
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:                                   # opcional
    LGBMClassifier = None

S = C.SEMILLA


def _pipe(modelo, escalar):
    pasos = ([("scaler", StandardScaler())] if escalar else []) + [("clf", modelo)]
    return Pipeline(pasos)


def catalogo_modelos(scale_pos_weight=1.0):
    """Modelos con manejo de desbalance incorporado (class_weight / scale_pos_weight)."""
    m = {
        "Regresión logística": _pipe(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=S), True),
        "KNN": _pipe(KNeighborsClassifier(n_neighbors=15), True),
        "SVM (RBF)": _pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=S), True),
        "Árbol de decisión": _pipe(DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=S), False),
        "Random Forest": _pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                      min_samples_leaf=3, n_jobs=-1, random_state=S), False),
        "Extra Trees": _pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                                  min_samples_leaf=3, n_jobs=-1, random_state=S), False),
        "Gradient Boosting": _pipe(GradientBoostingClassifier(random_state=S), False),
        "MLP": _pipe(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=800,
                                   early_stopping=True, random_state=S), True),
    }
    if XGBClassifier is not None:
        m["XGBoost"] = _pipe(XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                                           eval_metric="logloss", random_state=S, n_jobs=-1, verbosity=0), False)
    if LGBMClassifier is not None:
        m["LightGBM"] = _pipe(LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                             class_weight=None,
                                             random_state=S, n_jobs=-1, verbose=-1), False)
    return m


def espacios_busqueda():
    """Distribuciones para RandomizedSearchCV (claves con prefijo 'clf__')."""
    esp = {
        "Regresión logística": {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"],
                                "clf__solver": ["liblinear"]},
        "Random Forest": {"clf__n_estimators": randint(150, 600), "clf__max_depth": [4, 6, 8, 12, None],
                          "clf__min_samples_leaf": randint(1, 15), "clf__max_features": ["sqrt", 0.3, 0.5]},
        "Extra Trees": {"clf__n_estimators": randint(150, 600), "clf__max_depth": [4, 6, 8, 12, None],
                        "clf__min_samples_leaf": randint(1, 15), "clf__max_features": ["sqrt", 0.3, 0.5]},
        "Gradient Boosting": {"clf__n_estimators": randint(100, 500), "clf__learning_rate": loguniform(1e-2, 0.3),
                              "clf__max_depth": randint(2, 6), "clf__subsample": uniform(0.6, 0.4)},
        "SVM (RBF)": {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)},
        "KNN": {"clf__n_neighbors": randint(5, 60), "clf__weights": ["uniform", "distance"]},
        "MLP": {"clf__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)], "clf__alpha": loguniform(1e-4, 1e-1)},
        "Árbol de decisión": {"clf__max_depth": randint(2, 10), "clf__min_samples_leaf": randint(2, 30)},
    }
    if XGBClassifier is not None:
        esp["XGBoost"] = {"clf__n_estimators": randint(100, 500), "clf__max_depth": randint(2, 7),
                          "clf__learning_rate": loguniform(1e-2, 0.3), "clf__subsample": uniform(0.6, 0.4),
                          "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 10)}
    if LGBMClassifier is not None:
        esp["LightGBM"] = {"clf__n_estimators": randint(100, 500), "clf__num_leaves": randint(4, 40),
                           "clf__learning_rate": loguniform(1e-2, 0.3), "clf__subsample": uniform(0.6, 0.4),
                           "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_samples": randint(5, 40)}
    return esp


# ------------------------------------------------------------------ métricas
def mejor_umbral(y, p):
    """Umbral que maximiza F1 de la clase 'sequía' (se elige en validación/OOF, nunca en test)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    if len(f1) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1))])


def metricas(y, p, umbral=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= umbral).astype(int)
    out = {
        "F1": f1_score(y, pred, zero_division=0),
        "Precisión": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "Bal.Acc": balanced_accuracy_score(y, pred),
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }
    if len(np.unique(y)) == 2:
        out["PR-AUC"] = average_precision_score(y, p)
        out["ROC-AUC"] = roc_auc_score(y, p)
    else:
        out["PR-AUC"] = out["ROC-AUC"] = np.nan
    return out


def matriz_confusion(y, p, umbral=0.5):
    return confusion_matrix(np.asarray(y).astype(int), (np.asarray(p) >= umbral).astype(int), labels=[0, 1])


def probs(modelo, X):
    """Probabilidad de la clase 1 (sequía)."""
    return modelo.predict_proba(X)[:, 1]


def oof_probs(pipeline, X, y, fechas, n_splits=5, gap_meses=0, cv=None):
    """Probabilidades 'out-of-fold' con validación cruzada temporal.
    Devuelve (p, mascara_cubierta): el primer bloque temporal nunca es 'test' en TimeSeriesSplit."""
    p = np.full(len(y), np.nan)
    for tr, te in (cv if cv is not None else splits_panel(fechas, n_splits, gap_meses)):
        m = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        p[te] = probs(m, X.iloc[te])
    return p, ~np.isnan(p)


class ClimatologiaClassifier(ClassifierMixin, BaseEstimator):
    """Frecuencia suavizada de sequía por punto y mes; referencia interpretable."""

    def __init__(self, fuerza_prior=5):
        self.fuerza_prior = fuerza_prior

    def fit(self, X, y):
        d = X[["punto", "mes"]].copy()
        d["y"] = np.asarray(y)
        self.prevalencia_ = float(d["y"].mean())
        g = d.groupby(["punto", "mes"])["y"].agg(["sum", "count"])
        self.tasas_ = (g["sum"] + self.fuerza_prior * self.prevalencia_) / (g["count"] + self.fuerza_prior)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = X[["punto", "mes"]].join(self.tasas_.rename("p"), on=["punto", "mes"])["p"]
        p = p.fillna(self.prevalencia_).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def comparar_modelos(modelos, Xtr, ytr, Xva, yva, ftr, n_splits=5, gap_meses=1, verbose=True):
    """Entrena cada modelo en train, evalúa en validación y en CV temporal sobre train."""
    filas, fit = [], {}
    for nombre, pipe in modelos.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = clone(pipe).fit(Xtr, ytr)
            pva = probs(m, Xva)
            pcv, cub = oof_probs(pipe, Xtr, ytr, ftr, n_splits, gap_meses)
        thr = mejor_umbral(ytr.values[cub], pcv[cub])
        met = metricas(yva, pva, thr)
        met_cv = metricas(ytr.values[cub], pcv[cub], thr)
        filas.append({"Modelo": nombre, **{k + " (val)": v for k, v in met.items()},
                      "PR-AUC (CV)": met_cv["PR-AUC"], "F1 (CV)": met_cv["F1"], "umbral": thr})
        fit[nombre] = m
        if verbose:
            print(f"  {nombre:22s} F1 val={met['F1']:.3f}  PR-AUC val={met['PR-AUC']:.3f}")
    res = pd.DataFrame(filas).set_index("Modelo").sort_values("PR-AUC (val)", ascending=False)
    return res, fit


# ------------------------------------------------------------------ líneas base
def lineas_base(dtr, dva, ytr_col, yva_col):
    """Devuelve DataFrame con las líneas base evaluadas en validación.
    - Mayoritaria: siempre 'no sequía'.
    - Persistencia: si hoy hay sequía (SPI3<=-1) predice sequía en t+h.
    - Climatología: probabilidad histórica de sequía de ese mes y punto en train."""
    filas = []
    yva = dva[yva_col].values
    filas.append({"Modelo": "Base: siempre 'no sequía'", **metricas(yva, np.zeros(len(yva)), 0.5)})
    filas.append({"Modelo": "Base: persistencia", **metricas(yva, dva["sequia_actual"].values, 0.5)})
    clim = dtr.groupby(["punto", "mes"])[ytr_col].mean().rename("p")
    pc = dva[["punto", "mes"]].join(clim, on=["punto", "mes"])["p"].fillna(dtr[ytr_col].mean()).values
    thr = mejor_umbral(dtr[ytr_col].values, dtr[["punto", "mes"]].join(clim, on=["punto", "mes"])["p"].values)
    filas.append({"Modelo": "Base: climatología", **metricas(yva, pc, thr)})
    return pd.DataFrame(filas).set_index("Modelo")

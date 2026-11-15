# 🌵 Predicción de sequía meteorológica en el Altiplano boliviano

Proyecto de Análisis de Datos: preprocesamiento, EDA, tratamiento de outliers (IQR, MAD, Isolation Forest),
comparación de 10 modelos de machine learning, ajuste de hiperparámetros, evaluación y despliegue (API + app web).

**Pregunta:** con los datos hasta el mes *t*, ¿habrá sequía (SPI-3 ≤ −1) en el mes *t+h* (h = 1, 2, 3)?
**Datos públicos:** NASA POWER (diario, 1981–2025, 8 puntos del Altiplano) y NOAA ONI (El Niño).

> ⚠️ Herramienta **académica**. No es un pronóstico oficial de SENAMHI. Los datos son satelitales/de reanálisis, no de pluviómetros.

---

## 1. Qué contiene este paquete

```
proyecto_sequia/
├── notebooks/proyecto_sequia_completo.ipynb   ← TODO el análisis con gráficos (empieza aquí)
├── src/                                       ← código reutilizable
│   ├── config.py      puntos, fechas, umbrales, particiones (todo lo ajustable)
│   ├── datos.py       descarga NASA POWER y ONI, limpieza, agregación mensual
│   ├── spi.py         cálculo del SPI (gamma mixta)
│   ├── features.py    variables, etiquetas, particiones temporales, outliers
│   ├── modelos.py     catálogo de modelos, búsqueda de hiperparámetros, métricas
│   ├── pipeline.py    funciones de alto nivel (dataset, entrenamiento final, artefacto)
│   ├── predecir.py    predicción en producción
│   └── sintetico.py   datos SIMULADOS para probar sin internet (¡no usar en el informe!)
├── scripts/entrenar_y_guardar.py              ← entrena y guarda el modelo sin abrir el notebook
├── app/main.py                                ← API REST (FastAPI)
├── app/streamlit_app.py                       ← app web con mapa y semáforo de riesgo
├── tests/test_basico.py                       ← pruebas automáticas
├── data/raw/  data/processed/  models/  reports/figuras/   (se llenan al ejecutar)
├── requirements.txt  requirements-deploy.txt  Dockerfile  README.md
```

---

## 2. Requisitos
- **Python 3.10 o superior** (el código se probó en Python 3.12).
- Conexión a internet **la primera vez** (descarga de NASA POWER y NOAA). Después se usa la copia guardada en `data/raw/`.
- ~2 GB libres (librerías) y 8 GB de RAM recomendados.

## 3. Instalación (una sola vez)

Abre una terminal dentro de la carpeta `proyecto_sequia`.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> Si PowerShell bloquea el script de activación: `Set-ExecutionPolicy -Scope Process Bypass` y vuelve a intentar.
> Si `shap` o `lightgbm` fallan al instalar, el proyecto sigue funcionando (son opcionales; se omiten automáticamente).

## 4. Comprobar que todo funciona (sin internet, 1 minuto)
Con datos **simulados** (solo para verificar la instalación):
```bash
# Windows PowerShell:   $env:SEQUIA_SINTETICO="1"; pytest -q tests
# macOS / Linux:
SEQUIA_SINTETICO=1 pytest -q tests
```
Debe decir `8 passed`. (Estas pruebas no tocan tus datos reales.)

## 5. Ejecutar el análisis completo con datos reales
```bash
jupyter lab
```
Abre `notebooks/proyecto_sequia_completo.ipynb` y usa **Kernel → Restart & Run All**.

- **Primera ejecución:** descarga ~8 puntos × 45 años de NASA POWER (unos minutos) y guarda `data/raw/power_diario.csv` y `data/raw/oni.csv`.
- **Duración total (estimada):** de 10 a 30 minutos según tu equipo. Para una prueba rápida baja las iteraciones de la búsqueda de hiperparámetros:
  `SEQUIA_N_ITER=5` (por defecto 25).
- **Salidas:** gráficos en `reports/figuras/`, tablas en `reports/*.csv`, tabla de modelado en `data/processed/` y el modelo en `models/artefacto_sequia.joblib`.

Para empezar de cero (volver a descargar): borra `data/raw/power_diario.csv` y `data/raw/oni.csv`.

## 6. Entrenar sin el notebook (opcional)
```bash
python scripts/entrenar_y_guardar.py                    # selección automática por horizonte
python scripts/entrenar_y_guardar.py --modelo "Random Forest"
```
Por defecto compara modelos con pliegues temporales de 2011–2018 y elige uno por horizonte. `--modelo` fuerza un mismo modelo para los tres horizontes; sus nombres válidos son los de la sección 9 del notebook.

## 7. Ejecutar el despliegue

> Primero debe existir `models/artefacto_sequia.joblib` (lo crea el notebook, sección 18, o el script anterior).

**App web (Streamlit):**
```bash
streamlit run app/streamlit_app.py
```
Se abre en http://localhost:8501: mapa con semáforo de riesgo, resumen por punto, selector de horizonte y consulta de coordenadas nuevas.
La app indica el modelo elegido para cada horizonte. Si h=3 usa climatología, muestra un aviso: sigue siendo una referencia académica, no una alerta independiente.

**API (FastAPI):**
```bash
uvicorn app.main:app --reload
```
Documentación interactiva: http://127.0.0.1:8000/docs
Ejemplos:
```
GET /puntos
GET /predict?punto=Oruro&horizonte=1
GET /predict?lat=-17.5&lon=-66.0&horizonte=2      (punto nuevo: más lento y menos fiable)
```
Ambas consultan NASA POWER en el momento (necesitan internet) y usan el mes **completo** más reciente disponible.

**Docker:**
```bash
docker build -t sequia .
docker run -p 8000:8000 sequia                                   # API
docker run -p 8501:8501 sequia streamlit run app/streamlit_app.py --server.address 0.0.0.0   # app
```

**Publicación gratuita (resumen):**
- *Streamlit Community Cloud*: sube el proyecto a GitHub (incluye `models/artefacto_sequia.joblib`), crea la app apuntando a `app/streamlit_app.py`. Usa `requirements-deploy.txt` (cópialo como `requirements.txt` en el repositorio de despliegue para que la instalación sea liviana).
- *Hugging Face Spaces* (Docker o Streamlit) y *Render* también sirven con el `Dockerfile`.
- Para actualizar cada mes, reentrena y vuelve a subir el `.joblib`.

---

## 8. Cómo está construido (para tu informe)
| Etapa | Decisión |
|---|---|
| Etiqueta | SPI-3 del mes *t+h* ≤ −1 (OMM). Periodo de referencia 1981–2010 |
| Limpieza | `-999`→NaN, valores físicamente imposibles→NaN, cobertura mensual ≥ 90 % |
| Outliers | IQR global vs. IQR por mes (con guarda p99), MAD, Isolation Forest; votos; **extremos reales se conservan** |
| Experimento | ninguno vs. `log1p` vs. winsorización IQR → se mide el efecto en PR-AUC/F1 |
| Variables | rezagos, acumulados, SPI-1/3/6, anomalías, estacionalidad, ONI (rezago 2–3), altitud |
| Particiones | Train ≤ 2012 · Validación 2013–2018 · evaluación histórica 2019–2025; pliegues finales de selección solo desde 2011, con recorte de etiquetas t+h |
| Modelos | 10 modelos exploratorios; selección distinta por horizonte y referencia climatológica si supera a los modelos |
| Desbalance | `class_weight` / `scale_pos_weight`; comparación con SMOTE dentro de cada pliegue |
| Hiperparámetros | `RandomizedSearchCV` exploratorio para h=1; comparación final por horizonte en pliegues posteriores a la referencia SPI 1981–2010 |
| Umbral | Elegido con predicciones *out-of-fold* de 2011–2018, sin usar las etiquetas del test |
| Evaluación | F1, precisión, recall, PR-AUC, ROC-AUC, Brier, calibración, IC bootstrap por fechas |
| Análisis | Por horizonte, punto, mes y fase ENSO; permutación, SHAP, ablación; caso histórico 2015-2016 |

Las métricas 2019–2025 ya fueron inspeccionadas durante el desarrollo. Para confirmar mejoras posteriores hace falta evaluación temporal anidada o meses nuevos. El cuaderno ejecutado de entrega está en `proyecto_sequia_completo.ipynb`; una copia de las salidas queda en `reports/proyecto_sequia_ejecutado.ipynb`.

Dos decisiones que conviene defender:
1. **Se eliminó Viacha** respecto al plan inicial: NASA POWER tiene una grilla de ~0,5° y Viacha cae en la misma celda que La Paz (series duplicadas). El cuaderno comprueba esto en la sección 2.3.
2. **No se usan latitud/longitud como predictores**: con 8 puntos identificarían al punto y favorecerían el sobreajuste (se deja solo la altitud). Es configurable en `src/config.py`.

## 9. Problemas frecuentes
| Síntoma | Causa / solución |
|---|---|
| `RuntimeError: No se pudo descargar ... de NASA POWER` | Sin internet o servicio saturado. Reintenta en unos minutos. |
| `No se pudo obtener el ONI` | NOAA cambió la página. Abre https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/oni/v6/ , guárdala como `data/raw/oni_manual.txt` (Ctrl+S, o copia y pega el texto de la tabla) y vuelve a ejecutar. |
| `No existe models/artefacto_sequia.joblib` | Ejecuta el notebook completo o `scripts/entrenar_y_guardar.py`. |
| La app muestra aviso rojo “datos SINTÉTICOS” | El modelo se entrenó en modo prueba. Borra `models/*.joblib` y `data/raw/*sintetico*`, y reentrena con datos reales. |
| El notebook es muy lento | `SEQUIA_N_ITER=5` y/o quita modelos pesados (SVM, MLP) del catálogo en `src/modelos.py`. |
| `ModuleNotFoundError: src` | Abre Jupyter desde la carpeta raíz del proyecto (o desde `notebooks/`). |
| Predicción con “mes más reciente” atrasado | NASA POWER y el ONI se publican con retraso; la app usa el último mes completo con datos. |

## 10. Qué NO debes afirmar
- Que es un pronóstico oficial o que “declara” una sequía.
- Que los resultados valen para todo Bolivia (se entrenó con 8 puntos del Altiplano).
- Que la lluvia es medida en tierra (son datos satelitales y de reanálisis).

## 11. Licencias y datos
NASA POWER y NOAA CPC son datos públicos; cita las fuentes en tu informe. El código de este proyecto puede reutilizarse libremente con fines académicos.

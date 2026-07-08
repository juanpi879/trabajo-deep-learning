# Simulador Mundial 2026

Proyecto de prediccion e inteligencia deportiva para simular el Mundial 2026 con Python, notebooks y Streamlit.

## Contenido

- Preparacion de datos historicos y features.
- Entrenamiento de modelos predictivos.
- Dashboard interactivo para grupos, eliminatorias y campeones probables.
- Simulaciones con winrate, conteo de campeones y grafico de barras.
- Bracket visual actualizado segun el top 10 de campeones probables.

## Estructura principal

- `app_streamlit.py`: dashboard principal.
- `requirements.txt`: dependencias del proyecto.
- `notebooks/`: notebooks del proceso completo.
- `artifacts/data/`: datos procesados usados por la app.
- `artifacts/models/`: modelos entrenados.
- `archive (3)`, `archive (4)`, `archive (5)`, `archive (6)`: datasets base.

## Ejecucion

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Ejecutar dashboard:

```powershell
streamlit run app_streamlit.py
```

Abrir:

```text
http://localhost:8501
```

## Flujo recomendado

1. Ejecutar `notebooks/01_preparacion_datos_features.ipynb`.
2. Ejecutar `notebooks/02_modelos_deep_learning.ipynb`.
3. Ejecutar `notebooks/03_simulador_dashboard.ipynb` si se desea regenerar el dashboard.
4. Ejecutar `app_streamlit.py` con Streamlit.

## Nota

El dashboard usa directamente el modelo de Deep Learning entrenado en `notebooks/02_modelos_deep_learning.ipynb`
(`artifacts/models/dashboard_model.keras` + `artifacts/models/model_bundle.pkl` con el scaler y las columnas de
features). No hay ningun modelo alterno: las predicciones del simulador salen de la misma red neuronal (MLP)
comparada y evaluada en el notebook 02.

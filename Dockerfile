FROM python:3.11-slim
WORKDIR /app
# LightGBM necesita libgomp
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY src ./src
COPY app ./app
COPY models ./models
# Datos de referencia pequeños (ONI) se descargan en tiempo de ejecución
RUN mkdir -p data/raw data/processed reports/figuras
EXPOSE 8000 8501
# API por defecto:  docker run -p 8000:8000 sequia
# App web:          docker run -p 8501:8501 sequia streamlit run app/streamlit_app.py --server.address 0.0.0.0
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

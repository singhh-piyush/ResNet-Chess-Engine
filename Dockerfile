FROM node:22-slim AS frontend
WORKDIR /build
COPY chess-frontend/package*.json ./
RUN npm ci
COPY chess-frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
RUN useradd -m -u 1000 app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY engine ./engine
COPY backend ./backend
COPY main.py ./
COPY models ./models
COPY --from=frontend /build/dist ./chess-frontend/dist
USER app
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

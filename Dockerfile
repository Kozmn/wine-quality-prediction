FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY api/ ./api/
COPY models/ ./models/
EXPOSE 8000
CMD ["python", "api/app.py"]
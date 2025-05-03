# ----- build stage -----
    FROM python:3.11-slim  AS base

    WORKDIR /app
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # system deps for Pillow (JPEG, zlib) and psycopg2
    RUN apt-get update && apt-get install -y --no-install-recommends \
            libjpeg-turbo-progs zlib1g libpq5 \
        && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    
    # copy the rest of the code
    COPY . .
    
    EXPOSE 8501
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
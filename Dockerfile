# Use a lightweight Python base image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY src/ src/
COPY app/ app/
COPY README.md .
COPY data/ data/

# Expose Streamlit default port
EXPOSE 8501

# Entrypoint to run the Streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.enableCORS=false"]

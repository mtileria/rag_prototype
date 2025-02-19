FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Copy only requirements to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy rest of the source code
COPY . .

# Expose the application port
EXPOSE 8080

# Command to run the FastAPI application with auto-reload enabled 
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080", "--reload"]

# Build the Docker image
# docker build -t fastapi_app_image .

# Run the Docker container with a volume
# docker run -d --name ragapp -p 8080:8080 -v /path/to/repo/project_rag:/app fastapi_app_image
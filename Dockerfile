FROM python:3.10-slim

WORKDIR /app

# Copy all files first
COPY . .

# Install pip dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode (makes imports work)
RUN pip install -e .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run uvicorn with proper module path
CMD ["uvicorn", "Credit_Risk_Modelling.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
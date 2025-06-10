FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

# Install ExcelAgent dependencies
RUN pip install -r requirements.txt

# Copy ExcelAgent code
COPY . .

EXPOSE 8005

CMD ["uvicorn", "excel:app", "--host", "0.0.0.0", "--port", "8005"]
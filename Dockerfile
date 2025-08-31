FROM python:latest
WORKDIR /project

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git
RUN apt update && apt upgrade -y
RUN apt install tesseract-ocr tesseract-ocr-rus -y

COPY src ./src
# no exposure
# EXPOSE 5000

CMD ["python", "./src/main.py"]
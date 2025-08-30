FROM python:latest
WORKDIR /project

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/openai/CLIP.git

COPY src ./src
# no exposure
# EXPOSE 5000

CMD ["python", "./src/main.py"]
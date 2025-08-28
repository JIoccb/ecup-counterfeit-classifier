FROM python:latest
WORKDIR /project

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
# no exposure
# EXPOSE 5000

# for security reasons, we add user with no admin priveleges
RUN useradd app
# we give user an ability to write anything to its folder
RUN chown -R app /project
USER app

CMD ["python", "./src/main.py"]
FROM python:3.11
EXPOSE 8080

WORKDIR /app



# ENTRYPOINT ["streamlit", "run"]

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
RUN pip install --no-cache-dir --upgrade streamlit
RUN pip install --no-cache-dir --upgrade  opencv-python-headless


COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "ui.py"]
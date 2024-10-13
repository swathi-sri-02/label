FROM python:3.11-slim

RUN apt-get update --fix-missing

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libglib2.0-dev \
    libopencv-dev \
    && apt-get clean
	
RUN apt-get -y install build-essential python3-dev
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install libpoppler-glib-dev


RUN apt-get install -y python3 python3-pip
RUN pip3 install streamlit paddleocr opencv-python

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip3 install --upgrade pip 
RUN pip3 install -r /code/requirements.txt

COPY . /code

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
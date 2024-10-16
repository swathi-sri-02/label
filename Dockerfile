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
RUN apt-get install poppler-utils

RUN apt-get install -y python3 python3-pip
RUN pip3 install streamlit paddleocr opencv-python

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev 
    
WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip3 install --upgrade pip 
RUN pip3 install openpyxl
RUN pip3 install -r /code/requirements.txt

COPY . /code

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
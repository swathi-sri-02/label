FROM python:3.11-slim

RUN apt-get update --fix-missing

RUN apt-get -y install build-essential python3-dev
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --upgrade pip 
RUN pip install -r /code/requirements.txt

COPY . /code

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
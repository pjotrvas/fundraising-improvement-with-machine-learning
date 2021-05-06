FROM python:3

WORKDIR /mlmodel

COPY mlmodel.py mlmodel.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "mlmodel.py"]
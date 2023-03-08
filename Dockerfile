FROM python:3.11
WORKDIR /app
RUN mkdir -p /app/
RUN mkdir -p /app/templates
RUN mkdir -p /app/static/images/
RUN mkdir -p /app/static/images/techstack/
RUN mkdir -p /app/static/images/dcgan/
RUN mkdir -p /app/static/images/ddpm/
## copy static images
COPY ./static/images/*.png /app/static/images/
COPY ./static/images/techstack/* /app/static/images/techstack/
## copy templates
COPY ./templates/* /app/templates/
## install reqs
COPY requirements.txt /app
RUN pip install --trusted-host pypi.python.org --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt
## copy and run webserver
COPY app.py /app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers=2", "--threads=2", "app:app"]
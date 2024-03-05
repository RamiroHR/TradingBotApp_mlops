FROM python:3.9-slim

# need it: streamlit image will not work from the root drectory
WORKDIR /app  

# ADD ./docker_files/streamlit/app.py ./docker_files/streamlit/requirements.txt ./
ADD ./streamlit/app.py ./docker_files/streamlit/requirements.txt ./

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8501

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "app.py"]

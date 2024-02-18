FROM python:3.9-slim

ADD ./requirements.txt ./main.py ./database_management_tools.py ./

RUN apt-get update \
    && apt-get install -y build-essential libssl-dev \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

VOLUME /data
VOLUME /models
VOLUME /src

CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]
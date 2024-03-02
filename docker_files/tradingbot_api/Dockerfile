FROM python:3.9-slim

ADD ./docker_files/tradingbot_api/requirements.txt ./docker_files/tradingbot_api/main.py ./
# ADD ./requirements_main_api.txt ./main.py ./database_management_tools.py ./

COPY ./docker_files/tradingbot_api/src ./src
# COPY ./src ./src

RUN apt-get update \
    && apt-get install -y build-essential libssl-dev \
    && pip install --no-cache-dir -r requirements.txt

# RUN apt-get update \
#     && apt-get install -y build-essential libssl-dev \
#     && pip install --no-cache-dir -r requirements_main_api.txt

EXPOSE 8000

VOLUME /data
VOLUME /models
# VOLUME /src

CMD ["uvicorn", "main:api", "--host", "0.0.0.0"]
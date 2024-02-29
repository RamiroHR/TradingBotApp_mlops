#!./bin/bash

### Navigate to the project root folder "TradingBotApp_mlops"


### build the images (specifying the Dockerfile location)
docker image build . -t ramirohr/tradingbot:1.0.2 -f docker_files/tradingbot_api/Dockerfile


### run docker container & link ports & mount volumes
### only Winwods users should include the -W flag, otherwise remove it.
docker container run --name bot_api -p 8000:8000 -d \
--volume "$(pwd -W)/data:/data" \
--volume "$(pwd -W)/models:/models" \
--volume "$(pwd -W)/src:/src" \
ramirohr/tradingbot:1.0.2

### launch the docker compose
# docker-compose up

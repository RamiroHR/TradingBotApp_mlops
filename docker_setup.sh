#!./bin/bash

## Navigate to the root folder of the github project


### build the images for each test (specifying the Dockerfile location)
#docker image build . -t ramirohr/tradingbot:0.0.8
docker image build . -t ramirohr/tradingbot:0.0.8 -f docker_files/tradingbot_api/Dockerfile
#docker image build . -t authentication_image:latest -f authentication_test/Dockerfile

### run docker container & link ports & mount volumes
docker container run --name bot_api -p 8000:8000 -d \
--volume "$(pwd -W)/data:/data" \
--volume "$(pwd -W)/models:/models" \
--volume "$(pwd -W)/src:/src" \
ramirohr/tradingbot:0.0.8

### launch the docker compose
# docker-compose up
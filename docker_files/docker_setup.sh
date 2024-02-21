#!./bin/bash

### build the images for each test (specifying the Dockerfile location)
docker image build . -t ramirohr/tradingbot:0.0.8
#docker image build . -t authentication_image:latest -f authentication_test/Dockerfile


### launch the docker compose
# docker-compose up
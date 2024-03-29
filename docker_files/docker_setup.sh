#!./bin/bash

### Navigate to the project root folder "TradingBotApp_mlops"



#############################################################################
########################## --- MAIN ROUTER API --- ##########################

### build the images (specifying the Dockerfile location)
docker image build . -t ramirohr/tradingbot:1.1.1 -f docker_files/tradingbot_api/Dockerfile


### run docker container & link ports & mount volumes
### only Winwods users should include the -W flag, otherwise remove it.
docker container run --name bot_api -p 8000:8000 -d \
--volume "$(pwd -W)/data:/data" \
--volume "$(pwd -W)/models:/models" \
#--volume "$(pwd -W)/src:/src" \
ramirohr/tradingbot:1.1.1

### launch the docker compose
# docker-compose up



#############################################################################
################ --- USERS CREDENTIALS MICRO-SERVICE API --- ################

## build image
docker image build . -t ramirohr/credentials_api:1.0.5 -f docker_files/users_credentials_api/Dockerfile

## run a container
docker container run --name credential_api -p 7000:7000 -d ramirohr/credentials_api:1.0.5



#############################################################################
################## --- USERS INTERFACE - STREAMLIT --- ######################

## build image
docker image build . -t ramirohr/streamlit_ui:1.0.1 -f docker_files/streamlit/Dockerfile

## run a container
docker container run --name streamlit_ui -p 8501:8501 -d ramirohr/streamlit_ui:1.0.1

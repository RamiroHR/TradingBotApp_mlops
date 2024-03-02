#!./bin/bash

##############################################################
### >>> run this script from the project's root folder <<< ###
##############################################################

### create the secret
$ kubectl apply -f kubernetes/tb_secret.yml

### creates the deployment
$ kubectl apply -f kubernetes/tb_deployment.yml

### creates the service
kubectl apply -f kubernetes/tb_service.yml


##############################################################

### The APIs can be queries at port 8000 and 8001 of the localhost
### TradingBot api  : http://localhost:8000/docs
### Credentials api : http://localhost:8001/docs
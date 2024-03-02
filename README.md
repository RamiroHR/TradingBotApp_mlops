# Project Description
**Trading Bot App:** This project was developped as the evaluation project for the MLOps cursus.

The project consist on creating, deploying and monitoring an AI application with a model that aims to assist the user in determining the optimal timing for market entry. 


# Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docker_files
    |   ├── docker_setup.sh         <- Commands to build docker images.
    |   ├── tradingbot_api          <- File to create docker image of the "router api".
    |   |    ├── Dockerfile
    |   |    ├── main.py
    |   |    ├── requirements.txt
    |   |    └── src
    |   └── users_credentials_api   <- File to create docker image of the microservice "credentials_api".
    |
    ├── kubernetes         <- Deployment, service, secret yml files for kubernetes deployment of pods.
    |
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

<br>

# Deploy & launch the API
From the project's root folder execute the following commands to deploy a set of pods containerizing the API and its microservices.  
```
bash kubernetes/kube_setup.sh
```

The API can be queried in the browser going to the OpenAPI url 'http://localhost:8000/docs'

<br>

# Quickstart

Follow the steps below to swiftly utilize your initial model:

1. `/price_hist` : get the price up to date
2. `/update_model_params` : create and record the parameters for your model
3. `/train_model` : train and record the model
4. `/prediction` : predict

# The model
The API runs a trained model, with customizable parameters, of the family KNeighborsClassifier from sklearn.  
By looking at specific features extracted from the asset price history the model outputs a response aiming to assist the user in determining the optimal timing for market entry.

## Features engineering
The features are composed of four (4) momemtum indicators:
* Momentum
* Relative Strength Index (RSI) 
* Triple Exponential Average (TRIX)
* Moving Average Convergence Divergence (MACD)

## Target engineering
The target is a binary value (0 or 1) calculated from the exponential moving average (EMA). It involves determining the relative change between the EMA value on a specific day (D) and the EMA value after a certain number of days (D+x). A threshold is then applied to define the ratio of 1.
The exponential moving average is used to smooth daily prices and reduce noise in the price serie.

## The parameters
### Features & Target
| parameter | default value | description |
| ------- | ------- | ------- |
| features_length | 7 (int) | Defines the shortest period over which features are calculated. | 
| features_factor | 10 (int) | This factor enhances the number of features by multiplying the 'features_length' by each integer in the range(1, features_factor). |
| target_ema_length | 7 (int) | Determines the period for the exponential moving average used to smooth daily prices. |
| target_diff_length | 14 (int) | Specify the number of days into the future to consider when calculating the target.
| target_pct_threshold | 0.2 (float) | Value within the range of [0,1], representing the expected ratio of '1' labels.

### Model
Also refer to sklearn documentation.
| parameter | default value | description |
| ------- | ------- | ------- |
| n_neighbors | 30 (int) | Number of neighbors to use for kneighbors queries. |
| weights | "uniform" (str) | {‘uniform’, ‘distance’} Weight function used in prediction.  |

### Cross-validation
| parameter | default value | description |
| ------- | ------- | ------- |
| n_splits | 5 (int) | Number of splits. |

<br>
<br>
<br>
<br>

# Installation for project development
This is neccessary only for collaborators and developper users who intend to further improve the model using this repository.  

Create a new environment from the requirements.txt file.  

* ```pip install virtualenv```  (if you don't already have virtualenv installed)
* ```virtualenv env``` to create the new environment called 'env' within the project directory ('env' is in .gitignore such there is no issues when synchronizing with GitHub) 
* ```source env/bin/activate``` or ```source env/Scripts/activate``` to activate the virtual environment.
* ```pip install -r requirements.txt``` to install the requirements in the current environment

<br>

# Run Notebooks
1. To run the notebook within the activated vitual environment 'env' setup the jupyter kernel: `python -m ipykernel install --user --name=env`  
2. Then launch jupyter notebook and select a notebook: `jupyter notebook`  
3. In the menu under the notebook name select _**kernel**_ > _**change kernel**_ > _**env**_

<br>

# Launch individual the APIs
Run the following command in a terminal from the directory where the source code of the desired API is located.  
```
uvicorn main:api --reload
```

Then you can query the API at the browser going to the OpenAPI url 'http://localhost:8000/docs'

<br>

# Docker containerized APIs
The essential steps to build the docker image and the run the containerized API are detailed in the docker_files/docker_setup.sh file.  
The docker_setup.sh can be directly executed in a bash terminal from the project root folder.

Otherwise the main commands can be manually executed in the terminal from the project root folder:  
1. **Get the Docker image** 
The Docker image can be found at the DockerHub as **ramirohr/tradingbot:1.0.2**, ready to pull.  
Otherwise build the image locally (specifying the Dockerfile location)
    ```
    docker image build . -t ramirohr/tradingbot:1.0.2 -f docker_files/tradingbot_api/Dockerfile
    ```

2. **run docker container & link ports & mount volumes**  
(only Winwods users should include the -W flag, otherwise remove it)  

    ```
    docker container run --name bot_api -p 8000:8000 -d \
    --volume "$(pwd -W)/data:/data" \
    --volume "$(pwd -W)/models:/models" \
    --volume "$(pwd -W)/src:/src" \
    ramirohr/tradingbot:1.0.2
    ```
3. **Query the containerized API**
The containerized api is available at http://localhost:8000/docs

All docker images have been shared in Dockerhub.
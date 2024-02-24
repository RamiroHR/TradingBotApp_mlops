TradingBotApp_mlops Version 1.2.0
==============================

Trading Bot App as MLOps project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docker_files
    |   ├── docker_setup.sh  <- Commands to build docker images and run docker compose.
    |   └── tradingbot_api
    |       └── Dockerfile 
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


# Installation

Create a new environment from the requirements.txt file.  

* ```pip install virtualenv```  (if you don't already have virtualenv installed)
* ```virtualenv env``` to create the new environment called 'env' within the project directory ('env' is in .gitignore such there is no issues when synchronizing with GitHub) 
* ```source env/bin/activate``` or ```source env/Scripts/activate``` to activate the virtual environment.
* ```pip install -r requirements.txt``` to install the requirements in the current environment


# Launch the API
Run the following command in a terminal  
```
uvicorn main:api --reload
```

Then you can query the API at the browser going to the OpenAPI url 'http://localhost:8000/docs'


# Docker containerized API
The essential steps to build the docker image and the run the containerized API are detailed in the docker_files/docker_setup.sh file.  
The docker_setup.sh can be directly executed in a bash terminal from the project root folder.

Otherwise the main commands can be manually executed in the terminal from the project root folder:  
1. **build the images (specifying the Dockerfile location)**
    ```
    docker image build . -t ramirohr/tradingbot:5.0.0 -f docker_files/tradingbot_api/Dockerfile
    ```
2. **run docker container & link ports & mount volumes**  
(only Winwods users should include the -W flag, otherwise remove it)  

    ```
    docker container run --name bot_api -p 8000:8000 -d \
    --volume "$(pwd -W)/data:/data" \
    --volume "$(pwd -W)/models:/models" \
    --volume "$(pwd -W)/src:/src" \
    ramirohr/tradingbot:5.0.0
    ```
3. **Query the containerized API**
The containerized api is available at http://localhost:8000/docs

TradingBotApp_mlops Version 1.2.0
==============================

Trading Bot App as MLOps project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
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


# Installation

Create a new environment from the requirements.txt file.  

* ```pip install virtualenv```  (if you don't already have virtualenv installed)
* ```virtualenv env``` to create the new environment called 'env' within the project directory ('env' is in .gitignore such there is no issues when synchronizing with GitHub) 
* ```source env/bin/activate``` or ```source env/Scripts/activate``` to activate the virtual environment.
* ```pip install -r requirements.txt``` to install the requirements in the current environment


# Launch the API
Run the following command in a terminal  
```
uvicorn main:api --reload
```

Then you can query the API at the browser going to the OpenAPI url 'http://localhost:8000/docs'

# Quickstart

Follow the steps below to swiftly utilize your initial model:

1. `/price_hist` : get the price up to date
2. `/update_model_params` : create and record the parameter for your model
3. `/train_model` : train and record the model
4. `/prediction` : predict

# The model
## Purpose of the model
The current model aims to assist the user in determining the optimal timing for market entry.

## Features engineering
The features are composed of four (4) momemtum indicators:
* Momentum
* Relative Strength Index (RSI) 
* Triple Exponential Average (TRIX)
* Moving Average Convergence Divergence (MACD)

## Target engineering
The target is a binary value (0 or 1) calculated based on the relative change between the exponential moving average on a given day (D) and the price after a certain number of days (x).
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

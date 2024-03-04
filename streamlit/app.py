import streamlit as st
import requests 
import os


###--- SIDEBAR SELECTION ---###

# Define the sidebar navigation
asset_list = {
    'BTC':'BTCUSDT',
    'ETH':'ETHUSDT'
}

asset = st.sidebar.selectbox(
"Which asset do you want to buy ?",
options=list(asset_list.keys()),
index=None,
)

interval_list = {
    'Short term (1h interval)':'1h',
    'Middle term (4h interval)':'4h',
    'Long term (1d interval)':'1d'
}

interval = st.sidebar.selectbox(
"What horizon of time ?",
options=list(interval_list.keys()),
index=None,
)


###---Class tdbotAPI---###

class tdbotAPI:
    '''This class allows to connect to the API and call the various endpoints'''
    def __init__(self, url=''):
        self.url = url if url != '' else 'http://localhost:8000'

        self.user_id = 'user_1'
        self.user_pass = 'u_one'
        self.user_auth = (self.user_id, self.user_pass)

        self.admin_id = 'admin_1'
        self.admin_pass = 'a_one'
        self.admin_auth = (self.admin_id, self.admin_pass)

    def update_price_hist(self, asset, interval):
        endpoint = 'update_price_hist'
        url = os.path.join(self.url, endpoint)

        params = {'asset': asset, 'interval': interval}
        headers = {'accept': 'application/json'}

        response = requests.put(url, params=params, headers=headers)

        output = {}
        output['status_code'] = response.status_code

        if response.status_code == 200:
            output['data'] = response.json()
        
        return output
    
    def check_actual_price_hist(self, asset, interval):
        endpoint = 'check_actual_price_hist'
        url = os.path.join(self.url, endpoint)

        params = {'asset': asset, 'interval': interval}
        headers = {'accept': 'application/json'}

        response = requests.get(url, params=params, headers=headers)

        output = {}
        output['status_code'] = response.status_code

        if response.status_code == 200:
            output['data'] = response.json()
        
        return output

    def get_prediction(self, asset, interval):
        endpoint = 'prediction'
        url = os.path.join(self.url, endpoint)

        params = {'asset': asset, 'interval': interval, 'model_name': 'model_test'}
        headers = {'accept': 'application/json'}
        auth = self.user_auth

        response = requests.get(url, params=params, headers=headers, auth=auth)

        output = {}
        output['status_code'] = response.status_code

        if response.status_code == 200:
            output['data'] = response.json()

        return output
    
    def get_parameters(self, asset, interval):
        '''
        GET THE ACTUAL PARAMETERS
        '''
        # Define the URL of the API endpoint
        endpoint = 'get_model_params'
        url = os.path.join(self.url, endpoint)

        # Define the headers
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        params= {'asset': asset, 
                 'interval': interval, 
                 'model_name': 'model_test'}
        auth = self.admin_auth

        # Make the PUT request
        response = requests.get(url, params=params, headers=headers, auth=auth)

        output = {}
        output['status_code'] = response.status_code

        # Check the response status code
        if response.status_code == 200:
            output['data'] = response.json()

        return output
    
    def update_parameters(self, asset, interval, data):
        '''
        UPDATE THE PARAMETERS
        '''

        # Define the URL of the API endpoint
        endpoint = 'update_model_params'
        url = os.path.join(self.url, endpoint)

        # Define the headers, params & auth
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        params= {'asset': asset, 
                 'interval': interval, 
                 'model_name': 'model_test'}
        auth = self.admin_auth

        # Make the PUT request
        response = requests.put(url, params=params, json=data, headers=headers, auth=auth)

        output = {}
        output['status_code'] = response.status_code

        # Check the response status code
        if response.status_code == 200:
            output['data'] = response.json()

        return output
    
    def train_model(self, asset, interval):
        '''
        TRAIN MODEL
        '''

        # Define the URL of the API endpoint
        endpoint = 'train_model'
        url = os.path.join(self.url, endpoint)

        # Define the headers, params & auth
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        params= {'asset': asset, 
                 'interval': interval, 
                 'model_name': 'model_test'}
        auth = self.admin_auth

        # Make the PUT request
        response = requests.put(url, params=params, headers=headers, auth=auth)

        output = {}
        output['status_code'] = response.status_code

        # Check the response status code
        if response.status_code == 200:
            output['data'] = response.json()
        else:
            output['detail'] = response['detail']

        return output

# create an instance to call the API in the streamlit
tdb = tdbotAPI()




def page_time_to_buy():

    st.title('Time to buy ?')

    if st.button("Get prediction"):
        # If the button is clicked, call your function

        st.write('Your selection:')
        st.write('* Asset:', asset)
        st.write('* Horizon:', interval, '(time interval used: ', interval_list[interval], ')')

        with st.status('Updating historical data'):
            update_price = tdb.update_price_hist(asset_list[asset], interval_list[interval])
            if update_price['status_code']==200:
                st.write('Data updated. Last: ', update_price['data']['last_close_date'])
            else:
                st.write('Error in updating the historical data.')

        with st.status('Get prediction'):
            pred = tdb.get_prediction(asset_list[asset], interval_list[interval])
            st.write(pred)
        
        st.write('You should probably ', pred['data'], ' ;)')
        


def parameters_page():
    #st.header('Parameters')
    st.markdown('[Parameters definition available on Github](https://github.com/RamiroHR/TradingBotApp_mlops)')
    st.write('Please adjust the parameters you want.')

    if asset is None:
        st.markdown('<p style="color:red;">Select your asset.</p>', unsafe_allow_html=True)
    if interval is None:
        st.markdown('<p style="color:red;">Select your interval.</p>', unsafe_allow_html=True)

    if asset is not None and interval is not None:
        actual_params = tdb.get_parameters(asset_list[asset], interval_list[interval]) # get actual parameters
        
        if actual_params['status_code'] == 404: # use default_parameters
            actual_params={}
            actual_params['data'] = {
                    "params_features_eng": {
                        "features_length": 7,
                        "features_factor": 10,
                        "target_ema_length": 7,
                        "target_diff_length": 12,
                        "target_pct_threshold": 0.2
                    },
                    "params_model": {
                        "n_neighbors": 30,
                        "weights": "uniform"
                    },
                    "params_cv": {
                        "n_splits": 5
                    }
                }
        new_params = {} # collect the new parameters

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('For features')
            st.write('Features engineering')
            params_f = actual_params['data']['params_features_eng']
            new_params['features_length'] = st.number_input('features_length', value=params_f['features_length'], step=1)
            new_params['features_factor'] = st.number_input('features_factor', value=params_f['features_factor'], step=1)
            new_params['target_ema_length'] = st.number_input('target_ema_length', value=params_f['target_ema_length'], step=1)
            new_params['target_diff_length'] = st.number_input('target_diff_length', value=params_f['target_diff_length'], step=1)
            new_params['target_pct_threshold'] = st.number_input('target_pct_threshold', value=params_f['target_pct_threshold'], step=0.1, min_value=0.1, max_value=0.9)

        with col2:
            st.subheader('For model')
            st.write('Model: KNN Classifier')
            params_m = actual_params['data']['params_model']
            new_params['model_n_neighbors'] = st.number_input('n_neighbors', value=params_m['n_neighbors'], step=1)
            weights_list = ['uniform','distance']
            new_params['model_weights'] = st.selectbox('weights', weights_list, index=weights_list.index(params_m['weights']))

        with col3:
            st.subheader('For cross-validation')
            params_cv = actual_params['data']['params_cv']
            new_params['cv_n_splits'] = st.number_input('n_splits', value=params_cv['n_splits'], step=1)
        

        if st.button("Record new parameters"):
            with st.status('Update parameters'):
                response = tdb.update_parameters(asset_list[asset], interval_list[interval], new_params)
                if response['status_code']==200:
                    st.write('Parameters successfully updated')
                else:
                    st.write('Error. Parameters not updated.')

def training_page():
    st.write('Once your parameters are recorded. You can train the model for a given asset and horizon of time.')

    if not asset:
        st.markdown('<p style="color:red;">Warning - Please select an <b>asset</b> in the left panel</p>', unsafe_allow_html=True)
    elif not interval:
        st.markdown('<p style="color:red;">Warning - Please select an <b>horizon of time</b> in the left panel</p>', unsafe_allow_html=True)
    elif tdb.get_parameters(asset_list[asset], interval_list[interval])['status_code'] !=200: # check if the parameters exists
        st.markdown('<p style="color:red;">Warning - The parameters for '+asset_list[asset]+'-'+interval_list[interval]+' does not exist. Please go back to step 1 and record them.</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green;">You are ready to train the model</p>', unsafe_allow_html=True)

        if st.button("Train the model"):
            with st.status('Update the data'):
                response = tdb.update_price_hist(asset_list[asset], interval_list[interval])
                if response['status_code']==200:
                    st.write('Data successfully updated.')
                    st.write(response['data'])
                else:
                    st.write(response)

            with st.status('Train model'):
                response = tdb.train_model(asset_list[asset], interval_list[interval])
                if response['status_code']==200:
                    st.write('Model successfully trained')
                else:
                    st.write(f'Error {response["status_code"]}.')
        
            if response['status_code']==200:
                acc = round(response['data']['accuracy'],2)
                st.metric('Accuracy', value=acc)
                esc = round(response['data']['entry_score']*100,2)
                st.metric('Entry score', value=esc)


tab1, tab2, tab3 = st.tabs(['Step1 - Parameters', 'Step2 - Training', 'Step 3 - Time to buy ?'])

with tab1:
    parameters_page()

with tab2:
    training_page()

with tab3:
    page_time_to_buy()


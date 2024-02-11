import pandas as pd
import pandas_ta as ta
from sklearn.base import BaseEstimator, TransformerMixin

#--- CREATE CLASS GETFEATURES AND GETTARGET

class getFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, length, factor):
        '''
        Custom transformer to get the features

        Argument:
        * length    :   period used to calculate the indicators
        * factor    :   factor to be used to calculate the indicators on (length*factor) lengths
        '''
        self.length=length
        self.factor=factor

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        length_lead = self.length
        length_lag = self.length*2

        # calculate features
        C_=pd.DataFrame()
        for i in range(1,self.factor):
            length_lead = int(self.length*i)
            length_lag = length_lead*2
            C_['mom'+str(i)] = ta.mom(X, length=length_lead)
            C_['rsi'+str(i)] = ta.rsi(X, length=length_lead)
            C_['trix'+str(i)] = ta.trix(X, length=length_lead).iloc[:,0]
            C_['macd'+str(i)] = ta.macd(X, fast=length_lead, slow=length_lag).iloc[:,0]

        return C_
    
class getTarget(BaseEstimator, TransformerMixin):
    def __init__(self, length, pct_threshold=0.05):
        '''
        Custom transformer to get the target.

        Argument:
        * length    :   period used to calculate the indicators
        '''
        self.length=length
        self.pct_threshold = pct_threshold

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        length_lead = self.length
        length_lag = self.length*2

        # calculate target
        C_=pd.DataFrame()
        C_['ema'] = ta.ema(X, length=length_lead)
        C_['target'] = C_['ema'].pct_change(-length_lag) # if <0, price is likely to pump - if >0, price is likely to dump

        # create the target
        y_ = C_['target']
        y_=y_.apply(lambda x: 1 if x<-self.pct_threshold else 0)

        return y_


# CREATE PREPROCESSING FUNCTION


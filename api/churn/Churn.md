# Churn Class

```python
import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Churn( object ):
    
    def __init__( self ):
            self.home_path='/Users/user/repos/churn'
            self.credit_score_scaler  = pickle.load(open('/Users/user/Repos/churn/parameters/credit_score_scaler.pkl', 'rb'))
            self.age_scaler  =          pickle.load(open('/Users/user/Repos/churn/parameters/age_scaler.pkl', 'rb'))
            self.estimated_salary_scaler =  pickle.load(open('/Users/user/Repos/churn/parameters/estimated_salary_scaler.pkl', 'rb'))
            self.tenure_scaler =  pickle.load(open('/Users/user/Repos/churn/parameters/tenure_scaler.pkl', 'rb'))
            self.balance_scaler  = pickle.load(open('/Users/user/Repos/churn/parameters/balance_scaler.pkl', 'rb'))
            self.model = pickle.load(open('/Users/user/Repos/churn/src/model.pkl', 'rb'))
        

    def data_preparation( self, data ):
        
            #transform new columns
            cols_old = data.columns.to_list()
            snakecase = lambda x: inflection.underscore(x)
            cols_new = list(map(snakecase, cols_old))

            #rename
            data.columns = cols_new
            
            #drop columns
            data.drop(['customer_id','surname', 'exited', 'row_number'], axis=1, inplace=True)

            # Gender Encoding
            data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

            #Geography encoding
            data = pd.get_dummies( data, prefix=['geography'], columns=['geography'] )

            #Credit Score
            data['credit_score'] = self.credit_score_scaler.transform(data[['credit_score']].values)

            #Age
            data['age'] = self.age_scaler.transform(data[['age']].values)

            #Estimated Salary
            data['estimated_salary'] = self.estimated_salary_scaler.transform(data[['estimated_salary']].values)

            #Tenure
            data['tenure'] = self.tenure_scaler.transform(data[['tenure']].values)

            #Balance
            data['balance'] = self.balance_scaler.transform(data[['balance']].values)
            
                                                        
            return data                                            
            
    
    def get_prediction( self, model, original_data, data ):
        
            # prediction
            pred = model.predict( data )

            # join pred into the original data
            original_data['prediction'] = pred

            return original_data.to_json( orient='records', date_format='iso' )
```



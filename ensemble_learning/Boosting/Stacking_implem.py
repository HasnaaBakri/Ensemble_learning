import numpy as np 
import os
from sklearn.model_selection import train_test_split
os.system("cls")

class Stacking:
    def __init__(self , base_models , final_model):
        self.base_models=base_models
        self.final_model=final_model

    def fit(self , X, y):
        X_train , X_valid , y_train , y_valid=train_test_split(X,y  , test_size=0.4 , random_state=42)

        base_predictions=[]
        for model in self.base_models:
            model.fit(X_train , y_train)
            base_predictions.append(model.predict(X_valid))
        np.column_stack(base_predictions)
        self.final_model.fit(base_predictions , y)
    
    def predict(self , X):
        base_predictions=np.column_stack([model.predict(X) for model in self.base_models])
        return self.final_model.predict(base_predictions  )
        

            

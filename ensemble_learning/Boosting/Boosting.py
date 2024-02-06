import numpy as np 
import os
os.system("cls")
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

class AdaBoostClas:
    def __init__(self , n_estimators):
        self.n_estimators=n_estimators
        self.models=[]
        self.alphas=[]
        self.sample_weights=None
    def fit(self , X , y):
        n_samples , n_features=X.shape
        self.sample_weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            model=DecisionTreeClassifier(max_depth=1)
            model.fit(X,y,sample_weight=self.sample_weights)
            y_pred=model.predict(X)
            #error=np.sum(self.sample_wights*(y!=y_pred))/np.sum(self.sample_wights)
            error = np.sum(self.sample_weights * (y != y_pred)) / np.sum(self.sample_weights)
            alpha=0.5*np.log((1-error)/error)
            self.sample_weights*=np.exp(-y*y_pred *alpha)
            self.models.append(model)
            self.alphas.append(alpha)
    def predict(self, X):
        weak_pred=[model.predict(X) for model in self.models]
        weak_pred = np.array(weak_pred)
        self.alphas = np.array(self.alphas)
        weighted_pred = self.alphas.dot(weak_pred)
        return np.sign(weighted_pred)





class GradientBoostingReg :
    def __init__(self , n_estimators=150  , learning_rate=0.1 , max_depth=2):
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.models=[]
        self.y=None

    def fit(self , X, y):
        self.y=y
        initial_pred=np.mean(y)
        predicted_y=np.ones_like(y)*initial_pred
        for _ in range(self.n_estimators):
            error=y-predicted_y
            model=DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X,error)
            predicted_error=model.predict(X)
            predicted_y+=self.learning_rate*predicted_error
            self.models.append(model)
    
    def predict(self , X):
        predicted_y= np.mean(self.y)
        for model in self.models:
            predicted_y+=self.learning_rate*model.predict(X)
        return predicted_y
        





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[4]:


df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], 
                  columns=iris['feature_names'] + ['target'])
df


# In[5]:


X = df.iloc[:, [0, 2]]
y = df.iloc[:, [4]]


# In[6]:


from sklearn.model_selection import train_test_split
from flask import Flask, request
import pickle


# In[7]:


class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = None
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
    
    def predict(self, X):
        return np.where(self._net_input(X) >= 0, 1, -1)
    
    def _net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


# In[9]:


app = Flask(__name__)

#127.0.0.1:5000/api/predict?sl=4.5&pl=3.2
@app.route("/api/predict", methods=["GET"])
def home():
    model = Perceptron(0.01, 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    model.fit(X_train, y_train)
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    array = np.array([sepal_length, petal_length])
    prediction = model.predict(array)
    prediction = prediction.tolist()
    if prediction == 0:
        result="Iris Setosa"
    elif prediction == 1:
        result="Iris Versicolor"
    else:
        result="Iris Virginica"
    return result
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


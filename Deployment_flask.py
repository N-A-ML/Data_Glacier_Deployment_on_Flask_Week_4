#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, render_template
import joblib
from joblib import load
from sklearn.neighbors import KNeighborsClassifier

app=Flask(__name__)
model=load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features=[int(x) for x in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    output=round(prediction[0])
    return render_template('index.html', prediction_text='This iris flower is {}'.format(output))
if __name__=="__main__":
    app.run(port=5000)


# In[9]:


#get_ipython().system('jupyter nbconvert Deployment_flask.ipynb --to script')


# In[ ]:





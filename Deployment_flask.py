#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, render_template
import joblib
from joblib import load
from sklearn.neighbors import KNeighborsClassifier
import os
images_folder=os.path.join('static', 'images')
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = images_folder
model=load('model.joblib')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    pred_round=round(prediction[0])
    output=""
    if pred_round==0:
        output+="Setosa"
        file = os.path.join(app.config['UPLOAD_FOLDER'], 'setosa.jpg')
    elif pred_round==1:
        output+="Versicolor"
        file = os.path.join(app.config['UPLOAD_FOLDER'], 'versicolor.jpg')
    else:
        output+="Virginica"
        file = os.path.join(app.config['UPLOAD_FOLDER'], 'virginica.jpg')

    return render_template('index.html', prediction_text='This iris flower is {}'.format(output),
                           iris=file
                          )
if __name__=="__main__":
    app.run(port=5000, debug=True, use_reloader=False)


# In[16]:


#get_ipython().system('jupyter nbconvert Deployment_flask.ipynb --to script')


# In[ ]:





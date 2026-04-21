from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
model=pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route("/")
def about():
    return render_template('home.html')

@app.route("/home")
def about1():
    return render_template('home.html')

@app.route("/predict")      
def home1():
    return render_template('predict.html')

@app.route('/pred',methods=['POST','GET'])
def pred():
    x=[[obj for obj in request.form.values()]]
    x=np.array(x)
    output=model.predict(x)
    return render_template ("result.html",pred =str(output[0]) )

if __name__ == '__main__':
    app.run(debug=True)
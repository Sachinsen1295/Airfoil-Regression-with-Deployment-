import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS
import numpy as np
import pandas as pd


app = Flask(__name__)

model = pickle.load(open('Random_forest.sav','rb'))
@app.route("/home")
def home():
    # return 'Hello world'
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output= model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_feature = [np.array(data)]
    print(data)
   #  new_data = [list(data.values())]
    output= model.predict(final_feature)[0]
    print(output)
    return render_template('home.html',prediction_text = "Airfoil pressure {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
    
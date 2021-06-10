import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('Social_Network_Ads')
scFeatures = pickle.load(open('scFeatures.ft','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    rdSpend = float(request.form['rdSpend'])
    admSpend = float(request.form['admSpend'])

    finalFeatures = np.array([[rdSpend,admSpend]])
    stdFeatures = scFeatures.transform(finalFeatures)
    prediction = model.predict(stdFeatures)

    

    return render_template('index.html', prediction_text='Expected Profit from the Startup is  $ {}'.format(round(prediction[0][0])))


if __name__ == "__main__":
    app.run(debug=True)
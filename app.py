import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = tf.keras.models.load_model('SalaryPredictor')
scFeatures = pickle.load(open('FeatureTransformer.ft','rb'))

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

    finalFeatures = np.concatenate((np.array([[rdSpend,admSpend]])) , axis = 1)
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text='Expected Profit from the Startup is  $ {}'.format(round(prediction[0][0])))


if __name__ == "__main__":
    app.run(debug=True)
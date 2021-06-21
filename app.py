import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()] #[int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)
    predictions = prediction[:,1]
    THRESHOLD=0.206
    preds = np.where(predictions > THRESHOLD, 1, 0)
    output = preds[0]
    # if preds[0]==1:
    #     output = "FRAUD"
    # else:
    #     output = "NON FRAUD"

    return render_template('after.html', data=output)


if __name__ == "__main__":
    app.run(debug=True)
















from flask import Flask,render_template,request
import pickle
import numpy as np



model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_winequality():
    fixedacidity = float(request.form.get('fixed_acidity'))
    volatileacidity = float(request.form.get('volatile_acidity'))
    citricacid = float(request.form.get('citric_acid'))
    residualsugar = float(request.form.get('residual_sugar'))
    chlorides = float(request.form.get('chlorides'))
    freesulfurdioxide = float(request.form.get('free_sulfur_dioxide'))
    totalsulfurdioxide = float(request.form.get('total_sulfur_dioxide'))
    density = float(request.form.get('density'))
    pH = float(request.form.get('pH'))
    sulphates = float(request.form.get('sulphates'))
    alcohol= float(request.form.get('alcohol'))


    # prediction
    result = model.predict(np.array([fixedacidity,volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide,totalsulfurdioxide, density, pH,sulphates, alcohol]).reshape(1,11))
    # return str(result)
    if result[0] == 0:
        result = 'Bad'
    else:
        result="Good"

    return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(debug=True)
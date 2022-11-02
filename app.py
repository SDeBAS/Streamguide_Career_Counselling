
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/homepage')
def homepage():
    return render_template("homepage.html")


@app.route('/about')
def about():
    return render_template("aboutus.html")


@app.route('/test')
def test():
    return render_template("test.html")


@app.route('/contact')
def contact():
    return render_template("contacts.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        i = 0
        print(result)
        res = result.to_dict(flat=True)
        print("res:", res)
        arr1 = res.values()
        arr = ([value for value in arr1])

        data = np.array(arr)

        data = data.reshape(1, -1)
        print(data)
        loaded_model = pickle.load(open("model.pkl", 'rb'))
        predictions = loaded_model.predict(data)
       # return render_template('testafter.html',a=predictions)

        print(predictions)

        '''
         pred = loaded_model.predict_proba(data)
        print(pred)
        # acc=accuracy_score(pred,)
        pred = pred > 0.05
        # print(predictions)
        i = 0
        j = 0
        index = 0
        res = {}
        final_res = {}
        while j < 3:
            if pred[i, j]:
                res[index] = j
                index += 1
            j += 1
        # print(j)
        # print(res)
        index = 0
        for key, values in res.items():
            if values != predictions[0]:
                final_res[index] = values
                print('final_res[index]:', final_res[index])
                index += 1
        # print(final_res)
        jobs_dict = {0: 'Science',
                     1: 'Commerce',
                     2: 'Arts',
                     }

        # print(jobs_dict[predictions[0]])
        job = {}
        # job[0] = jobs_dict[predictions[0]]
        index = 1

        data1 = predictions[0]
        print(data1)
         if predictions == 1:
            final='Science'
        if predictions[0][1] == 1:
            final='Commerce'
        if predictions[0][2] == 1:
            final='Arts'
        '''
       

        return render_template("results.html",final=predictions[0])


if __name__ == "__main__":
    app.run(debug=True)

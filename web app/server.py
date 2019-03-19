from flask import Flask, request,jsonify
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

def detecting_fake_news(ip):    
#retrieving the best model for prediction call
    load_model = pickle.load(open('final_model.sav', 'rb'))
    prediction = load_model.predict([ip])
    prob = load_model.predict_proba([ip])
    return jsonify(
        statement=ip,
        result=str(prediction[0]),
        probability = prob[0][1]
        )
    # return (print("The given statement is ",prediction[0]),
    #     print("The truth probability score is ",prob[0][1]))
@app.route('/hello')
@cross_origin()
def hello():
    return "Hello"

@app.route('/news',methods=['POST'])
@cross_origin()
def verify():
    if request.method == 'POST':
      article = request.form['article']
      return detecting_fake_news(article)

if __name__ == '__main__':
   app.run(port=5000)
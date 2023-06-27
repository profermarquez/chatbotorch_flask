from flask import Flask
from flask import request,render_template
from flask_cors import CORS, cross_origin
from flask import jsonify
import os
import sys
import json

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='127.0.0.1', port=port)

@app.route('/')
@cross_origin()
def index():
    #return jsonify({'ip': request.remote_addr}), 200
    return "<p><a href=""/chat"">Chat</a></p> <p><a href=""/backend"">Backend</a></p>"

@app.route('/train')
@cross_origin()
def train():
    my_dir='D:\Escuelas\Campus de Robotica\2023\ECO9 2023\openia_python\chatbotorch_gradio\static\train.py'

    """os_res=os.system('%s %s' % (sys.executable,os.path.join(my_dir, 'train.py')))
    print(os_res) """
    ver = "python "+my_dir
    # This method will return the exit status of the command
    status = os.system(ver)
    print('The returned Value is: ', status)
    
    return "train"

@app.route('/backend')
@cross_origin()
def projects():
    return render_template("index.html", title = 'Frontend')

@app.route('/chat')
@cross_origin()
def chat():
    return render_template("chat.html")

@app.route('/json', methods=['POST'])
@cross_origin()
def hello():
    data = request.json
    
    #data = {data}
    #print("previo  ",data)
    with open('static/intents.json','w',encoding='utf-8') as j:
        json.dump(data,j)
        #print("json ",j)

    """ home_dir = os.system("py train.py")
    print(home_dir) """
    return 'cargado'
    
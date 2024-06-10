import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from Perceptron.sakura_perceptron import train
from Perceptron.sakura_perceptron import SakuraPerceptron
from Perceptron.data_loader import load_X_Y,load_data,norm_data
import numpy as np
import torch
import torch.nn as nn

import pandas as pd
import eventlet.green.threading as threading
import os


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# # file handlering
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

thread_event = threading.Event()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode=None)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS        
@app.route("/")
def index():
   return render_template("homepage.html")

@app.route('/uploadData', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        # filesize = os.stat(file_path)
        try:
            df = pd.read_csv(file_path)
            rows = df.shape[0]
            columns = df.columns[:]
            return jsonify({
                'message': 'File successfully uploaded',
                'file_info': {
                    'name': filename,
                    'rows': rows,
                    'columns': [*columns]
                }
            }), 200
        except Exception as e:
            return jsonify({'message': 'Error reading CSV file: {}'.format(str(e))}), 400
    else:
        return jsonify({'message': 'Only CSS files are allowed'}), 400

@app.route('/start_training', methods=['POST'])
# @app.route('/start_training')
def start_background_program():
    # data = request.json
    # print(data)
    print("background program start")
    try:
        if not thread_event.set():
            thread_event.set()
            train_data_path = os.path.join(UPLOAD_FOLDER, "train_data.csv")
            train_data_label_path = os.path.join(UPLOAD_FOLDER, "train_data_label.csv")
            eventlet.spawn(train, socketio, train_data_path, train_data_label_path, batch_size=32, learning_rate=0.01)
        
        # return jsonify(data)
        return "",200
    except Exception as e:
        return str(e)
  
# stop background program
@app.route('/stop_training', methods=['POST'])
def stop_background_program():
   try:
      thread_event.clear()
      return "background program was stoped"
   except Exception as e:
      return str(e)

# @app.route("/validation")
# def start_validation():
#     test_data_path = "Data/test_data.csv"
#     test_data_label_path = "Data/test_data_label.csv"
#     true_values, predicted_values = validation(test_data_path, test_data_label_path)
#     return render_template('validation.html', true_values=true_values, predicted_values=predicted_values)

# def validation(test_data_path, test_data_label_path):
#     model = SakuraPerceptron()
#     model.load_state_dict(torch.load("sakura_model_second.pt"))
#     model.eval()

#     data_raw, label = load_data(test_data_path, test_data_label_path)
#     data_raw = norm_data(data_raw, "avg_temp")
#     X, Y_true = load_X_Y(data_raw, label)
#     X = torch.tensor(X, dtype=torch.float32)

#     Y_predict = []

#     with torch.no_grad():
#         for x in X:
#             output = model(x)
#             Y_predict.append(output.item())

#     return np.array(Y_true), np.array(Y_predict)


# test connect
@socketio.on('my event')
def handel_event(json):
    print(json)
    
if __name__ == '__main__':
    socketio.run(app, debug=True)
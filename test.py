# app.py

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from threading import Thread
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Create some initial data
x = np.linspace(0, 10, 100)
y = np.sin(x)
source = ColumnDataSource(data=dict(x=x, y=y))

# Create Bokeh plot
# Create Bokeh plot
plot = figure(title="Real-time data", height=300, width=600, y_range=(-1.5, 1.5))
plot.line('x', 'y', source=source, line_width=2)

# Function to update data
def update_data():
    while True:
        new_x = np.linspace(0, 10, 100)
        new_y = np.sin(new_x)
        source.data = dict(x=new_x, y=new_y)
        socketio.emit('update_plot', {'x': new_x.tolist(), 'y': new_y.tolist()})
        socketio.sleep(1)  # Update data every second

# Start a background thread to continuously update data
thread = Thread(target=update_data)
thread.daemon = True
thread.start()

# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index1.html')

# Socket.IO event handler
@socketio.on('connect')
def handle_connect():
    emit('update_plot', {'x': source.data['x'].tolist(), 'y': source.data['y'].tolist()})

if __name__ == '__main__':
    socketio.run(app)

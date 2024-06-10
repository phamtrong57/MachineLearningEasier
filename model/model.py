import time
import random
from flask_socketio import SocketIO

socketio = SocketIO()

def train_model():
    for epoch in range(100):  # Simulate 10 epochs
        loss = random.uniform(0.1, 1.0)  # Simulate a random loss value
        print("Running")
        socketio.emit('update_loss', {'epoch': epoch + 1, 'loss': loss})
        time.sleep(1)  # Simulate time taken for an epoch


from flask import Flask, render_template, request, redirect, jsonify, url_for, flash, session
from flask_socketio import SocketIO
from werkzeug.security import check_password_hash
import time
import random
import sqlite3

# Import the database initialization and user management functions
from database import init_db, add_user, get_user

# Import the file handling functions
from file_handler import handle_file_upload

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
socketio = SocketIO(app, async_mode = None)

"1.<--------------------------------------- login and resgiter>"


@app.route('/')
def home():
    if 'username' in session:
        # return f'Logged in as {session["username"]}'
        return render_template('index.html')
    # return 'You are not logged in <br><a href="/login">Login</a>'
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = get_user(username)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
        elif get_user(username):
            flash('Username already exists', 'danger')
        else:
            try:
                add_user(username, password)
                flash('Registration successful! You can now login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Registration failed. Please try again.', 'danger')
    
    return render_template('register.html')


"1.<-------------------------------------------- login and resgiter>"


"2.<-------------------------------------------- file upload and model config"

@app.route('/upload', methods=['POST'])
def upload_file():
    if handle_file_upload(request):
        return jsonify({'message': 'File successfully uploaded'}), 200
    else:
        return jsonify({'message': 'Allowed file types are csv'}), 400

@app.route('/configure_model', methods=['POST'])
def configure_model():
    input_neurons = request.form['input_neurons']
    hidden_layers = request.form['hidden_layers']
    neurons_per_layer = request.form['neurons_per_layer']
    # Save or process model configuration as needed
    flash(f'Model configured with {input_neurons} input neurons, {hidden_layers} hidden layers, and {neurons_per_layer} neurons per hidden layer.', 'success')
    return redirect(url_for('training'))

@app.route('/training')
def training():
    return render_template('training.html')
"2."

"3. ------------------------ Training and display"

"3. ------------------------ Training and display"
if __name__ == '__main__':
   socketio.run(app, debug=True)

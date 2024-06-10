from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/test")
def test():
    return "<p> test </p>"

@app.route('/test_user/<username>')
def test_user(username):
    return f'{username}, Hello World'
#!/usr/bin/env python3
"""
Simple test server for Analysis Zoo template
"""

from flask import Flask, render_template
import os

app = Flask(__name__, 
           template_folder='../../web/templates',
           static_folder='../../web/static')

@app.route('/')
def index():
    return '<h1>NeuronMap Test Server</h1><p><a href="/zoo">Analysis Zoo</a></p>'

@app.route('/zoo')
def zoo():
    return render_template('zoo_test.html')

if __name__ == '__main__':
    print("🧪 Starting Analysis Zoo Test Server...")
    print("📍 URL: http://localhost:5003/zoo")
    app.run(debug=True, host='127.0.0.1', port=5003)

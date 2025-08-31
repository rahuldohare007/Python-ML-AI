from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('../model/classifier.h5')
class_names = ['cat', 'dog', 'flower']  # update according to dataset

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            path = os.path.join('static/uploaded', img_file.filename)
            img_file.save(path)
            img = Image.open(path).resize((150, 150))
            img_array = np.expand_dims(np.array(img)/255.0, axis=0)
            pred = model.predict(img_array)
            prediction = class_names[np.argmax(pred)]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

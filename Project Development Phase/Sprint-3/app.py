import os

import numpy as np
from flask import Flask, render_template, request, send_from_directory, url_for
#from gevent.pywsgi import WSGIServer
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from werkzeug.utils import redirect, secure_filename

UPLOAD_FOLDER = 'D:/NalaiyaThiran/projFiles/data'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("./model/mnist_digit_recog_cnn.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/web', methods=['GET', 'POST'])
def web():
    if request.method == "POST":
        f = request.files["image"]
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'data', f.filename)
        f.save(filepath)
        # img = image.load_img(filepath, target_size=(64, 64))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)

        # filepath = secure_filename(f.filename)
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        # upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img = Image.open(filepath).convert("L")  # convert image to monochrome
        img = img.resize((28, 28))  # resizing of input image

        im2arr = np.array(img)  # converting to image
        im2arr = im2arr.reshape(1, 28, 28, 1)  # reshaping according to our requirement

        pred = model.predict(im2arr)

        num = np.argmax(pred, axis=1)  # printing our Labels

        return render_template('web.html', num=str(num[0]))
    return render_template('web.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
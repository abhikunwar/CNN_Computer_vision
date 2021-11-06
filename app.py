from flask import Flask, render_template
from PIL import Image
import base64
import io
import keras
from keras.preprocessing.image import load_img, img_to_array



app = Flask(__name__)
@app.route('/')
def first():
    return render_template("first.html")

@app.route('/submit')
def hello_world():

    # Full Script.

    image = r'E:\ineuron_learning\tfod1_web_app\dog_cat\data\validation\cats\cat.2002 - Copy.jpg'

    im = Image.open(image)
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())



    model = keras.models.load_model('models/first_ann_2021_11_05_14_07_06.h5')
    img = load_img(image, target_size=(64, 64))
    img_arr = img_to_array(img)
    img_arr = img_arr.reshape(1, 64, 64, 3)
    data = model.predict_classes(img_arr)
    print('class',data)

    return render_template("index.html", img_data=encoded_img_data.decode('utf-8'),data = data)


if __name__ == '__main__':
    app.run()
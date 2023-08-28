from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load CSV labels
csv_label = pd.read_csv("labels.csv")
labels = csv_label["breed"].to_numpy()
unique_breed = np.unique(labels)

# Load pre-trained model
full_model = tf.keras.models.load_model('20230828-07031693206192-full-image.h5',
                                       custom_objects={"KerasLayer": hub.KerasLayer})

IMG_SIZE = 224

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

def create_data_batches(x, batch_size=32):
    data_batch = preprocess_image(x)
    data_batch = tf.expand_dims(data_batch, axis=0)
    return data_batch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = 'temp_image.jpg'
            uploaded_file.save(image_path)
            test_batch = create_data_batches(image_path)
            predictions = full_model.predict(test_batch)
            predicted_breed = unique_breed[np.argmax(predictions)]
            return render_template('index.html', prediction_text=f'Name of the Dog : {predicted_breed}')
        else:
            return "No file uploaded"
    except Exception as e:
        return "An error occurred: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)

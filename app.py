from flask import Flask, jsonify, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image
import io
import base64
import cv2
import hashlib

app = Flask(__name__, template_folder='data/template')

# Load the model
model = load_model('data/model.h5')

# Define the class labels
diseases = ['cataract', 'conjuctivitis', 'glaucoma', 'normal']

# Define a simple XOR encryption key
encryption_key = b'my_secret_key'

@app.route('/', methods=['POST', 'GET'])
def process_image():
    if request.method == 'POST':
        # Load and preprocess the image
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        test_input = img.reshape(1, 224, 224, 3)

        # Get the probabilities for each class
        probabilities = model.predict(test_input)[0]

        # Get the index of classes with probabilities greater than 50%
        predicted_classes = np.where(probabilities > 0.5)[0]

        # Prepare the prediction result
        prediction_result = []
        for i in predicted_classes:
            disease = diseases[i]
            probability = probabilities[i] * 100
            prediction_result.append({'disease': disease, 'probability': probability})

        # Generate LIME Explanation
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0  # Normalize pixel values

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_array[0].numpy(), model.predict, top_labels=5, hide_color=0, num_samples=50)

        # Display LIME Explanation
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
        lime_img = mark_boundaries(temp / 2 + 0.5, mask)
        lime_img[mask] = [0, 1, 0]  # Set positive regions to green (RGB)

        # Convert images to base64 and encrypt
        original_img_base64 = encrypt_data(image_to_base64(img), encryption_key)
        lime_img_base64 = encrypt_data(image_to_base64(Image.fromarray((lime_img * 255).astype(np.uint8))), encryption_key)

        # Create a hash value and encrypt it
        hash_value = hashlib.sha256((original_img_base64 + lime_img_base64).encode('utf-8')).hexdigest()
        encrypted_hash = encrypt_data(hash_value, encryption_key)

        return render_template('result_bloc.html', predictions=prediction_result, original_img_base64=original_img_base64, lime_img_base64=lime_img_base64, encrypted_hash=encrypted_hash, encryption_key=encryption_key.decode('utf-8'))

    return render_template('result_bloc.html')

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encrypt_data(data, key):
    encrypted = bytearray(data.encode('utf-8'))
    for i, byte in enumerate(encrypted):
        encrypted[i] ^= key[i % len(key)]
    return base64.b64encode(encrypted).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_classification_model(model_path):
    return load_model(model_path)

def predict_image_class(model, image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
                    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    return class_labels[predicted_class[0]]

# Paths
model_path = r'C:\Repository\Staar project\code\satellite_image_classification_model.h5'
image_path = r'C:\Repository\Staar project\code\SeaLake_198.jpg'

# Load model
model = load_classification_model(model_path)

# Predict
predicted_class = predict_image_class(model, image_path)
print(f"Predicted class: {predicted_class}")

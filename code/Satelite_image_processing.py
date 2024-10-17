import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Path to the dataset
data_dir = r'/mnt/c/Repository/Staar project/eurosat'

# Initialize lists to hold images and labels
images = []
labels = []

# Iterate over each class directory
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    # Ensure that the path is a directory
    if os.path.isdir(class_dir):
        print(f"Processing directory: {class_dir}")
        # Iterate over each file in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Check if the path is a file and has a valid image extension
            if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    # Open, convert, resize, and store the image
                    img = Image.open(img_path).convert('RGB').resize((224, 224))
                    images.append(np.array(img))
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
labels_onehot = to_categorical(labels_encoded, num_classes=num_classes)

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_onehot, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the model with transfer learning (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Ensure num_classes matches

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard callback
tensorboard_callback = TensorBoard(log_dir='./logs')

def generator(image_paths, labels, batch_size):
    while True:
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            batch_images = [np.array(Image.open(p).convert('RGB').resize((224, 224))) for p in batch_paths]
            batch_images = np.array(batch_images).astype('float32') / 255.0
            yield batch_images, batch_labels

# Train the model
history = model.fit(
    generator(X_train_paths, y_train, batch_size),
    validation_data=generator(X_val_paths, y_val, batch_size),
    steps_per_epoch=len(X_train_paths) // batch_size,
    validation_steps=len(X_val_paths) // batch_size,
    epochs=10,
    callbacks=[tensorboard_callback]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')
print(f'Test Loss: {test_loss}')

# Save the model
model.save('satellite_image_classification_model.h5')

# Load the model (when needed)
loaded_model = tf.keras.models.load_model('satellite_image_classification_model.h5')

# Predict on test data
y_pred = np.argmax(loaded_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print(confusion_matrix(y_true, y_pred))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

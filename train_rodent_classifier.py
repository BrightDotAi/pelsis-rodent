import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# Create directories for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Define dataset paths
data_dir = "data"
rodent_dir = os.path.join(data_dir, "rodent")
not_rodent_dir = os.path.join(data_dir, "not_rodent")

# Function to load and preprocess images
def load_images_from_folder(folder, label, img_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='rgb', target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

# Load rodent and not_rodent images
rodent_images, rodent_labels = load_images_from_folder(rodent_dir, label=1)
not_rodent_images, not_rodent_labels = load_images_from_folder(not_rodent_dir, label=0)

# Combine datasets
X = np.concatenate([rodent_images, not_rodent_images])
y = np.concatenate([rodent_labels, not_rodent_labels])

# Split into training (65%), validation (15%), and test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.57, random_state=42)  # 0.57 * 0.35 ≈ 0.20

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# No augmentation for validation and test sets
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Load pre-trained MobileNetV2 model (without the top classification layer)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Train the model (transfer learning)
print("Training the model with transfer learning...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=16),
    steps_per_epoch=len(X_train) // 16,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=16),
    validation_steps=len(X_val) // 16,
    epochs=50,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

# Fine-tune the model
print("Fine-tuning the model...")
# Unfreeze the top layers of the base model
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

# Continue training
history_fine = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=16),
    steps_per_epoch=len(X_train) // 16,
    validation_data=val_datagen.flow(X_val, y_val, batch_size=16),
    validation_steps=len(X_val) // 16,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

# Evaluate on the test set
test_loss, test_acc, test_auc = model.evaluate(test_datagen.flow(X_test, y_test, batch_size=16))
print(f'Test Accuracy: {test_acc}, Test AUC: {test_auc}')

# Save evaluation metrics to a text file
with open("results/evaluation_metrics.txt", "w") as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    f.write(f"Test AUC: {test_auc}\n")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Transfer Learning Accuracy')
plt.plot(history_fine.history['accuracy'], label='Fine-Tuning Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Transfer Learning Loss')
plt.plot(history_fine.history['loss'], label='Fine-Tuning Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot to a file
plt.savefig("results/training_history.png")
plt.show()

# Save the TensorFlow model
model.save("models/rodent_classifier.h5")
print("TensorFlow model saved to models/rodent_classifier.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("models/rodent_classifier.tflite", "wb") as f:
    f.write(tflite_model)
print("TensorFlow Lite model saved to models/rodent_classifier.tflite")

print("Model training and conversion to TensorFlow Lite completed!")
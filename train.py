import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

if tf.test.is_gpu_available():
    print("GPU is available. However, we will restrict to CPU as requested.")
    tf.config.set_visible_devices([], 'GPU')
else:
    print("GPU is not available. Using CPU.")

# Confirm that only CPU is visible
print(f"Visible devices: {tf.config.get_visible_devices()}")

import shutil
from sklearn.model_selection import train_test_split

# Image dimensions for AlexNet-like model
img_height, img_width = 224, 224

# Define the root directory where your original 'Ring' and 'NonRing' folders are
source_root_dir = 'Galaxy'

# Define the base directory for the structured dataset (where train/val will be created)
dest_base_data_dir = os.path.join(source_root_dir, 'data')

train_dir = os.path.join(dest_base_data_dir, 'train')
val_dir = os.path.join(dest_base_data_dir, 'val')

# Define class directories within the source
class_folders = ['Ring', 'NonRing']

# --- Create the target directory structure ---
os.makedirs(os.path.join(train_dir, 'Ring'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'NonRing'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'Ring'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'NonRing'), exist_ok=True)

print(f"Structuring dataset from '{source_root_dir}' to '{dest_base_data_dir}'...")

# --- Split and copy images ---
for class_name in class_folders:
    source_class_path = os.path.join(source_root_dir, class_name)
    if not os.path.exists(source_class_path):
        print(f"Warning: Source folder '{source_class_path}' not found. Please ensure your images are in 'Galaxy/Ring' and 'Galaxy/NonRing'.")
        continue

    all_images = [os.path.join(source_class_path, img) for img in os.listdir(source_class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not all_images:
        print(f"Warning: No images found in '{source_class_path}'. Please check the folder content.")
        continue

    # Split images into training and validation sets
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    # Copy training images
    for img_path in train_images:
        shutil.copy(img_path, os.path.join(train_dir, class_name, os.path.basename(img_path)))

    # Copy validation images
    for img_path in val_images:
        shutil.copy(img_path, os.path.join(val_dir, class_name, os.path.basename(img_path)))

print("Dataset structuring complete.")

batch_size = 32 # You can adjust this

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data preprocessing for validation/test (minimal augmentation, just rescaling)
test_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
)

print("Loading data from structured directories...")

# First pass to get initial counts and handle imbalance
# Use a basic generator to just count samples before augmentation for imbalance handling
initial_train_generator = test_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=1, # Small batch size for counting
    class_mode='categorical',
    shuffle=False # Don't shuffle for consistent counting
)

class_counts = {class_name: 0 for class_name in initial_train_generator.class_indices.keys()}

# Create a mapping from class index to class name using class_indices
idx_to_class_name = {v: k for k, v in initial_train_generator.class_indices.items()}

for class_idx in initial_train_generator.classes:
    class_name = idx_to_class_name[class_idx]
    class_counts[class_name] += 1

print(f"Initial training class counts: {class_counts}")

# Identify minority/majority classes
min_class = min(class_counts, key=class_counts.get)
max_class = max(class_counts, key=class_counts.get)

if class_counts[min_class] < class_counts[max_class]:
    print(f"Imbalance detected. Minority class: '{min_class}' ({class_counts[min_class]} samples).")
    target_count = class_counts[max_class] # Balance to match majority class
    needed_augmentations = target_count - class_counts[min_class]

    if needed_augmentations > 0:
        print(f"Generating {needed_augmentations} augmented images for '{min_class}'...")
        # Create a generator specifically for augmenting the minority class
        # pointing only to the minority class folder
        minority_class_train_dir = os.path.join(train_dir, min_class)
        
        # Define an aggressive ImageDataGenerator for minority class augmentation
        minority_augment_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Use flow method to generate and save augmented images
        augment_generator = minority_augment_datagen.flow_from_directory(
            train_dir, # Point to the overall train_dir, but only process minority class
            target_size=(img_height, img_width),
            batch_size=1, # Generate one by one
            class_mode='categorical',
            classes=[min_class], # Only generate for the minority class
            save_to_dir=os.path.join(train_dir, min_class), # Save into the minority class folder
            save_prefix='aug',
            save_format='jpeg'
        )

        # Generate the needed number of images
        for i in range(needed_augmentations):
            next(augment_generator)
        print("Augmentation complete for minority class.")

# Now, create the final train_generator which includes the original and newly augmented images
train_generator = train_datagen.flow_from_directory(
    train_dir, # Path to the training directory (now balanced)
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    val_dir, # Path to the validation directory
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Keep data in order for evaluation metrics (important for confusion matrix)
)

# Re-evaluate num_classes and class_names in case augmentation changed things (unlikely but safe)
num_classes = train_generator.num_classes # Dynamically set num_classes
class_names = list(train_generator.class_indices.keys())

dataset_sizes = {
    'train': train_generator.samples,
    'val': validation_generator.samples
}

print(f"Updated number of training samples: {dataset_sizes['train']}")
print(f"Number of validation samples: {dataset_sizes['val']}")
print(f"Classes: {class_names}")

def create_alexnet(input_shape=(img_height, img_width, 3), num_classes=10):
    model = Sequential([
        # First Convolutional Block
        Conv2D(filters=64, kernel_size=(5, 5), strides=(4, 4), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        # Second Convolutional Block
        Conv2D(filters=192, kernel_size=(5, 5), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        # Third Convolutional Block
        Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
        # Fourth Convolutional Block
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        # Fifth Convolutional Block
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        # Flatten for fully connected layers
        Flatten(),
        # First Fully Connected Layer
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Second Fully Connected Layer
        Dense(4096, activation='relu'),
        Dropout(0.5),
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    return model

# Initialize the model
model = create_alexnet(num_classes=num_classes)

model.summary()

# Optimizer
# Using SGD with learning rate and momentum as in the PyTorch example
optimizer = SGD(learning_rate=0.001, momentum=0.9)

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Scheduler (Keras Callbacks)
# Decay LR by a factor of 0.1 every 7 epochs
def lr_scheduler(epoch, lr):
    if epoch % 7 == 0 and epoch != 0:
        return lr * 0.1
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

num_epochs = 80 # Increased epochs for better training, you can adjust this

print("Starting model training...")
history = model.fit(
    train_generator, # Use the prepared ImageDataGenerator for training
    epochs=num_epochs,
    validation_data=validation_generator, # Use the prepared ImageDataGenerator for validation
    callbacks=[lr_callback]
)

print("Training complete.")

# Optionally save the trained model
model.save('alexnet_galactic_rings_tf.h5')
print("Model saved to alexnet_galactic_rings_tf.h5")

print("Evaluating model on test data...")
loss, accuracy = model.evaluate(validation_generator)
print(f'Accuracy of the network on the test images: {accuracy*100:.2f}%')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

print("Generating predictions for detailed evaluation...")

# Get predictions on the validation set
validation_generator.reset() # Important to reset generator before predicting
Y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("galaxy_val_cm.png")

# Classification Report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# AUC Score (for binary classification)
# If num_classes > 2, AUC for each class needs to be calculated in a One-vs-Rest fashion
if num_classes == 2:
    # Assuming 'Ring' is the positive class (index 0 or 1)
    # Adjust if your 'Ring' class is not at class_names[0]
    ring_class_idx = class_names.index('Ring') if 'Ring' in class_names else 0 # Assuming 'Ring' is the positive class
    y_scores = Y_pred[:, ring_class_idx]

    auc = roc_auc_score(y_true, y_scores)
    print(f'AUC Score: {auc:.4f}')

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
elif num_classes > 2:
    print("AUC calculation for multi-class classification (One-vs-Rest):")
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_scores = Y_pred[:, i]
        auc = roc_auc_score(y_true_binary, y_scores)
        print(f'  AUC for {class_name}: {auc:.4f}')

from tensorflow.keras.preprocessing import image as keras_image
import random

def predict_single_image(model, img_path, class_names, target_size=(img_height, img_width)):
    # Load and preprocess the image
    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array /= 255.0 # Normalize

    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    return predicted_class, confidence, img_array[0] # Return the processed image for display

# --- Specify your image path here ---
# For example:
custom_image_path = 'Galaxy/Ring/119.709500_9.589100.jpeg' # <--- CHANGE THIS TO YOUR IMAGE PATH

if not os.path.exists(custom_image_path):
    print(f"Warning: The specified image path '{custom_image_path}' does not exist.")
    print("Selecting a random image from the validation set instead.")

    # Get a random image path from the validation set for demonstration
    all_val_image_paths = []
    for class_name_val in class_names:
        class_folder = os.path.join(val_dir, class_name_val)
        if os.path.exists(class_folder):
            images_in_folder = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_val_image_paths.extend(images_in_folder)

    if not all_val_image_paths:
        print("No images found in the validation directory. Cannot perform single image prediction.")
    else:
        example_img_path = random.choice(all_val_image_paths)
        # Determine the actual class name from the path for display purposes
        actual_class_name = os.path.basename(os.path.dirname(example_img_path))

        # Make a prediction
        predicted_class, confidence, display_img = predict_single_image(model, example_img_path, class_names, (img_height, img_width))

        # Display the image and prediction
        plt.imshow(display_img)
        plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})\nActual: {actual_class_name}")
        plt.axis('off')
        plt.show()

        print(f"The model predicts the image is a '{predicted_class}' with a confidence of {confidence:.2f}.")
        print(f"The actual class is '{actual_class_name}'.")

else:
    # Use the specified custom image path
    example_img_path = custom_image_path
    actual_class_name = os.path.basename(os.path.dirname(example_img_path)) # Infer actual class from folder name

    # Make a prediction
    predicted_class, confidence, display_img = predict_single_image(model, example_img_path, class_names, (img_height, img_width))

    # Display the image and prediction
    plt.imshow(display_img)
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})\nActual: {actual_class_name}")
    plt.axis('off')
    plt.show()

    print(f"The model predicts the image is a '{predicted_class}' with a confidence of {confidence:.2f}.")
    print(f"The actual class is '{actual_class_name}'.")

def predict_single_image(model, image, class_names):
    # Expand dimensions to create a batch of 1 image
    image = np.expand_dims(image, axis=0)
    # Preprocess (resize and normalize) the image
    image = tf.image.resize(image, (img_height, img_width))
    image = image / 255.0 # Ensure normalization matches training

    predictions = model.predict(image)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    return predicted_class, confidence

# Get a random image from the test set (original un-preprocessed for display)
# We need to load CIFAR-10 again to get un-preprocessed images for display
(_, _), (raw_x_test, raw_y_test) = tf.keras.datasets.cifar10.load_data()

random_idx = np.random.randint(len(raw_x_test))
example_input_raw = raw_x_test[random_idx]
example_label_raw = raw_y_test[random_idx][0] # Get integer label

# Make a prediction using the raw image
predicted_class, confidence = predict_single_image(model, example_input_raw, class_names)

# Display the image and prediction
plt.imshow(example_input_raw)
plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})\nActual: {class_names[example_label_raw]}")
plt.axis('off')
plt.show()

print(f"The model predicts the image is a '{predicted_class}' with a confidence of {confidence:.2f}.")
print(f"The actual class is '{class_names[example_label_raw]}'.")

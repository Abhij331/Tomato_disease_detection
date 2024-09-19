# IMPORTING LIBRARIES
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path='/kaggle/input/tomato-leaf-detections/tomato/train'
plt.figure(figsize=(70,70))
count=0
plant_names=[]
total_images=0
for i in os.listdir(path):
    count+=1
    plant_names.append(i)
    plt.subplot(7,7,count)

    images_path=os.listdir(path+"/"+i)
    print("Number of images of "+i+":",len(images_path),"||",end=" ")
    total_images+=len(images_path)

    image_show=plt.imread(path+"/"+i+"/"+images_path[0])
    plt.imshow(image_show)
    plt.xlabel(i)
    plt.xticks([])
    plt.yticks([])

print("Total number of images we have",total_images)
print(plant_names)
print(len(plant_names))

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Dense, Flatten, ZeroPadding2D, BatchNormalization, Activation, Add, Input, Dropout, GlobalAveragePooling2D, Layer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

class CastLayer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

base_model_tf = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

# Model building
base_model_tf.trainable = False

pt = Input(shape=(224,224,3))
x = CastLayer()(pt)
x = preprocess_input(x)  # This function used to zero-center each color channel wrt Imagenet dataset
x = base_model_tf(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(len(plant_names), activation='softmax')(x)  # Ensure the number of classes matches your dataset

model_main = Model(inputs=pt, outputs=x)
model_main.summary()

train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=False, vertical_flip=False,
                                   fill_mode='nearest', width_shift_range=0.2, height_shift_range=0.2)

val_datagen = ImageDataGenerator()

path_train = '/kaggle/input/tomato-leaf-detections/tomato/train'
path_valid = '/kaggle/input/tomato-leaf-detections/tomato/val'

train = train_datagen.flow_from_directory(directory=path_train, batch_size=32, target_size=(224,224),
                                          color_mode='rgb', class_mode='categorical', seed=42)

valid = val_datagen.flow_from_directory(directory=path_valid, batch_size=32, target_size=(224,224), color_mode='rgb', class_mode='categorical')

# Callbacks
es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=7, mode='auto')
mc = ModelCheckpoint(filepath='/content/best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, patience=5, min_lr=0.001)

model_main.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_main.fit(train, validation_data=valid, epochs=30, steps_per_epoch=200, verbose=1, callbacks=[mc, es, lr])

model_main.save("RESNET50_PLANT_DISEASE.h5")

# Plotting loss and accuracy
plt.figure(figsize=(10,5))
plt.plot(model_main.history.history['loss'], color='b', label='Training loss')
plt.plot(model_main.history.history['val_loss'], color='r', label='Validation loss')
plt.xlabel("epochs")
plt.ylabel("loss_value")
plt.title("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(model_main.history.history['accuracy'], color='b', label='Training accuracy')
plt.plot(model_main.history.history['val_accuracy'], color='r', label='Validation accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("Accuracy")
plt.legend()
plt.show()
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions
valid.reset()  # Reset the validation generator to start from the beginning
predictions = model_main.predict(valid, steps=valid.samples // valid.batch_size + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid.classes
class_labels = list(valid.class_indices.keys())

# Print classification report
print("Classification Report")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

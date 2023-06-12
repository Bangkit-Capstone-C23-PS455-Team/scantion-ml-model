#Libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

#Load Image HAM10000 Datasets
base_dir = os.path.join('Dataset')

benign_dir = os.path.join(base_dir, 'train','benign')
malignant_dir = os.path.join(base_dir, 'train', 'malignant') 
normal_dir = os.path.join(base_dir, 'train','normal')

print('total training benign images:', len(os.listdir(benign_dir)))
print('total training malignant images:', len(os.listdir(malignant_dir)))
print('total training normal images:', len(os.listdir(normal_dir)))

pic_index = 2

benign_files = os.listdir(benign_dir)
malignant_files = os.listdir(malignant_dir)
normal_files = os.listdir(normal_dir)

next_benign = [os.path.join(benign_dir, fname) 
                for fname in benign_files[pic_index-2:pic_index]]
next_malignant = [os.path.join(malignant_dir, fname) 
                for fname in malignant_files[pic_index-2:pic_index]]
next_normal = [os.path.join(normal_dir, fname) 
                for fname in normal_files[pic_index-2:pic_index]]

fig, axs = plt.subplots(1, len(next_benign + next_malignant + next_normal), figsize=(20,7))

for i, img_path in enumerate(next_benign + next_malignant + next_normal):
    img = mpimg.imread(img_path)
    axs[i].imshow(img)
    axs[i].axis('off')

model = Sequential([
    tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(optimizer = Adam(learning_rate=0.0001),
              loss = "categorical_crossentropy",
              metrics=["accuracy"])

#Image Augmented
from keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = os.path.join(base_dir, 'train')
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = os.path.join(base_dir, 'test')
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(100,100),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(100,100),
	class_mode='categorical', 
  batch_size=126
)

# Train the model
history = model.fit(train_generator, epochs=25, validation_data = validation_generator, verbose = 1)

loss, accuracy = model.evaluate(train_generator, verbose=1)
val_loss, val_acc = model.evaluate(validation_generator, verbose=1)
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
print("Validation: accuracy = %f  ;  loss = %f" % (val_acc, val_loss))
model.save("model.h5")
model.save_weights('weights.h5')

# Plot the results
import seaborn as sns
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
fig, ax = plt.subplots(1,2, figsize=(15, 5))
fig.text(s='Training and validation accuracy/loss', size=20, fontweight='bold',
          y=1, x=0.28,alpha=0.8)

sns.despine()
ax[0].plot(epochs, acc, 'r', label='Training accuracy')
ax[0].plot(epochs, val_acc, 'b', label='Validation accuracy')
ax[0].legend(loc=0)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs, loss, 'r', label='Training loss')
ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
ax[1].legend(loc=0)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Training & Validation Loss')

fig.show()

#Define Path
model_path = "model.h5"
model_weights_path = 'weights.h5'
input = 'data_test\ISIC_0024545.jpg' #input path

#Load model and weight
model = load_model(model_path)
model.load_weights(model_weights_path)

#Prediction Function

def img_pred(file):
    x = load_img(file, target_size=(100,100))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    classes = np.argmax(result)
    print(classes)
    if classes == 0:
        print("Predicted: Benign")
    elif classes == 1:
        print("Predicted: Malignant")
    elif classes == 2:
        print("Predicted: Normal")
    else:
        print("not predicted")
    return classes

predicted = img_pred(input)
print(" ")

from IPython.display import display
import ipywidgets as widgets
from ipywidgets import *

button = widgets.Button(description="Predict")
out = widgets.Output()
uploader = widgets.FileUpload()
def on_button_clicked(_):
    with out:
        clear_output()
        try:
            img_pred(uploader)
        except:
            print("no image uploaded")
display(uploader)

button.on_click(on_button_clicked)
widgets.VBox([button, out])

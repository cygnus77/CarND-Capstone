import os
import numpy as np
from glob import glob
import re
import scipy.misc
import fnmatch
import random
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import keras.applications.inception_v3
import keras.applications.vgg16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import TruncatedNormal
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

num_classes = 4
IMAGE_HT = 300
IMAGE_WD = 400
data_dir = '/home/anand/CarND-Capstone/data/'
runs_dir = './runs'
learning_rate = 0.001
batch_size = 40
epochs = 3

def count_pngs(dir):
  count = 0
  for root, dirnames, filenames in os.walk(dir):
    for filename in fnmatch.filter(filenames, '*.png'):
        count += 1
  return count

# create the base pre-trained model
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_HT, IMAGE_WD, 3) )
# optimizer = RMSprop(lr=learning_rate)
#input_preprocessor = keras.applications.inception_v3.preprocess_input
#FROZEN_LAYERS = 172

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_HT, IMAGE_WD, 3))
optimizer = Adam(lr=learning_rate)
#optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True)
input_preprocessor = keras.applications.vgg16.preprocess_input
FROZEN_LAYERS = 11

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# and a logistic layer --  we have 4 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers
for layer in base_model.layers:
  layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen =  ImageDataGenerator(
  preprocessing_function=input_preprocessor,
  width_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  fill_mode='nearest',
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  preprocessing_function=input_preprocessor
)

train_generator = train_datagen.flow_from_directory(
  os.path.join(data_dir, 'train'),
  target_size=(IMAGE_HT, IMAGE_WD),
  batch_size=batch_size,
  classes=["red", "yellow", "green", "nolight"],
  class_mode="categorical"
)
validation_generator = test_datagen.flow_from_directory(
  os.path.join(data_dir, 'val'),
  target_size=(IMAGE_HT, IMAGE_WD),
  batch_size=batch_size,
  classes=["red", "yellow", "green", "nolight"],
  class_mode="categorical"
)

nb_train = count_pngs(os.path.join(data_dir, 'train'))
nb_val = count_pngs(os.path.join(data_dir, 'val'))

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_loss', verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')


model.fit_generator(train_generator, nb_train // batch_size, epochs, 
  verbose=1, validation_data=validation_generator,
  validation_steps=nb_val // batch_size,
  class_weight={0:1., 1: 1., 2: 1., 3: 1.},
  use_multiprocessing=True,
  callbacks = [early, checkpoint])

# start fine-tuning conv layers in addition to FC. 
# freeze the bottom FROZEN_LAYERS layers, train the rest
for layer in model.layers[:FROZEN_LAYERS]:
   layer.trainable = False
for layer in model.layers[FROZEN_LAYERS:]:
   layer.trainable = True

# compile
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# training for fine-tuning conv layers
model.fit_generator(train_generator, nb_train // batch_size, epochs, 
  verbose=1, validation_data=validation_generator,
  validation_steps=nb_val // batch_size,
  class_weight='auto',
  use_multiprocessing=True,
  callbacks = [early, checkpoint])

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



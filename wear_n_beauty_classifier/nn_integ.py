import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras.applications as app
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from keras.models import load_model

from pathlib import Path

from class_indices import class_indices, true


class Classifier:
    def __init__(self):
        self.IMG_SHAPE = (150, 150, 3)
        self.base_model = app.EfficientNetV2M(input_shape=self.IMG_SHAPE, include_top=False)

    def extra_layers(self):
        self.model = tf.keras.Sequential([
            self.base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(121, activation='softmax')
        ])
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='Adam')


ozon = Classifier()
ozon.extra_layers()

images = Path('../images/')


class Data():
    def __init__(self):
        pass

    def prepare_data(preprocess_input, BATCH_SIZE=1, IMG_SHAPE=150, prod_dir=images):
        image_gen_prod = ImageDataGenerator(preprocessing_function=preprocess_input)
        prod_data_gen = image_gen_prod.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=prod_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='categorical')
        return prod_data_gen

data = Data()
prod_data_gen = Data.prepare_

model = load_model('../mnist-dense.hdf5')data(preprocess_input=preprocess_input, prod_dir=images)

pred = model.predict_generator(prod_data_gen)
pred_class_num = np.argmax(pred, axis=1)

if pred_class_num in true:
    pass
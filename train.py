import numpy as np  # linear algebra
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random
import gc
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    Input,
    GlobalAveragePooling2D,
    Concatenate,
    Activation,
)
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from data_generator import DataGenerator
from constant import (
    TRAIN_FILE_PATH,
    TRAIN_IMAGE_PATH,
    FOLD,
    BATCH_SIZE,
    DIM,
    SAVE_MODEL_PATH,
    EPOCHS
)
from learning_rate import step_decay
from model import build_model
from loss import custom_mse


df = pd.read_csv(TRAIN_FILE_PATH)
df["filepath"] = TRAIN_IMAGE_PATH
df.loc[df["speed"] > 1, "speed"] = 1
df_train = df[df["fold"] != FOLD].reset_index(drop=True)
df_all_test = df[df["fold"] == FOLD].reset_index(drop=True)

df_val, df_test = train_test_split(df_all_test, test_size=0.5, random_state=0)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

train_gen = DataGenerator(
    df_train, BATCH_SIZE, dimension=DIM, shuffle=True, augment=True
)
val_gen = DataGenerator(df_val, BATCH_SIZE, dimension=DIM, shuffle=False)

checkpoint = ModelCheckpoint(
    SAVE_MODEL_PATH, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

lrate = LearningRateScheduler(step_decay)

## Initial Train

STEP = len(df_train)//BATCH_SIZE
VAL_STEP = len(df_val)//BATCH_SIZE

model = build_model()

model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEP,
    validation_data=val_gen,
    validation_steps=VAL_STEP,
    callbacks=[
        checkpoint, 
        lrate,
    ],
    verbose = 1
)

del model
gc.collect()

model = tf.keras.models.load_model(
    SAVE_MODEL_PATH, 
    custom_objects={"custom_mse": custom_mse}
    )

## 1st finetune

LR = 1e-6
EPOCHS = 300

for layer in model.layers[7].layers[-4:-1]:
    layer.trainable = True

for layer in model.layers[3:5]:
    layer.trainable = True

model.layers[6].trainable = True
    
model.compile(
    Adam(learning_rate=LR), 
    loss=custom_mse, 
    metrics=['mse']
)

model.summary()

checkpoint = ModelCheckpoint(SAVE_MODEL_PATH, 
                                 monitor='val_loss', 
                                 verbose= 1,
                                 save_best_only=True, 
                                 mode= 'min')

model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEP,
    validation_data=val_gen,
    validation_steps=VAL_STEP,
    callbacks=[
        checkpoint, 
    ],
    verbose = 1
)

del model
gc.collect()

model = tf.keras.models.load_model(
    SAVE_MODEL_PATH,
    custom_objects={
        'custom_mse': custom_mse
        }
    )

##2nd finetune
LR = 1e-6/2
EPOCHS = 150

for layer in model.layers[7].layers[-8:-5]:
    layer.trainable = True
    
for layer in model.layers[1:3]:
    layer.trainable = True
    
model.compile(
    Adam(learning_rate=LR), 
    loss=custom_mse, 
    metrics=['mse']
)

print('--------------------------2nd unfreeze-------------------------')

model.summary()

checkpoint = ModelCheckpoint(SAVE_MODEL_PATH, 
                                 monitor='val_loss', 
                                 verbose= 1,
                                 save_best_only=True, 
                                 mode= 'min')

model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=STEP,
    validation_data=val_gen,
    validation_steps=VAL_STEP,
    callbacks=[
        checkpoint, 
    ],
    verbose = 1
)

del model
gc.collect()

model = tf.keras.models.load_model(
    SAVE_MODEL_PATH,
    custom_objects={
        'custom_mse': custom_mse
        }
    )

## Report result on validate and test set
print("load validata data")
X1_val = np.zeros((len(df_val), DIM[0], DIM[1], 3))
X2_val = np.zeros((len(df_val), 224, 224, 3))
for i in tqdm(range(len(df_val))):
    img_id = df_val["image_id"][i]
    image = cv2.imread(TRAIN_IMAGE_PATH + str(img_id) + ".png")
    image = image.astype(np.float32) / 255.0
    image1 = cv2.resize(image, (DIM[1], DIM[0]))
    image2 = cv2.resize(image, (224, 224))
    X1_val[i, :, :, :] = image1
    X2_val[i, :, :, :] = image2


X1_test = np.zeros((len(df_test), DIM[0], DIM[1], 3))
X2_test = np.zeros((len(df_test), 224, 224, 3))

for i in tqdm(range(len(df_test))):
    img_id = df_test["image_id"][i]
    image = cv2.imread(TRAIN_IMAGE_PATH + str(img_id) + ".png")
    image = image.astype(np.float32) / 255.0
    image1 = cv2.resize(image, (DIM[1], DIM[0]))
    image2 = cv2.resize(image, (224, 224))
    X1_test[i, :, :, :] = image1
    X2_test[i, :, :, :] = image2

y_val_pred = model.predict([X1_val, X2_val])
y_test_pred = model.predict([X1_test, X2_test])

y_val_pred[y_val_pred < 0] = 0
y_val_pred[y_val_pred > 1] = 1

y_test_pred[y_test_pred < 0] = 0
y_test_pred[y_test_pred > 1] = 1


print('-----------------------------------2nd unfreeze-------------------------------------------')

mse = tf.keras.losses.MeanSquaredError()
print(mse(df_val[['angle', 'speed']].values, y_val_pred).numpy())
print(mse(df_test[['angle', 'speed']].values, y_test_pred).numpy())
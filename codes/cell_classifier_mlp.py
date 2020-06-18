# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 0.001
N_NEURONS = (300, 500, 300)
N_EPOCHS = 50
BATCH_SIZE = 512
DROPOUT = 0.2

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load('x_train.npy'), np.load('y_train.npy')

# Augmentation by GAN-generated images
x_aug_type1 = np.load('gen_samples_type1.npy')
x_aug_type2 = np.load('gen_samples_type2.npy')
x_aug_type3 = np.load('gen_samples_type3.npy')
x_aug = np.append(x, x_aug_type1, axis=0)
x_aug = np.append(x_aug, x_aug_type2, axis=0)
x_aug = np.append(x_aug, x_aug_type3, axis=0)
y_aug = np.append(y, np.ones(1000))
y_aug = np.append(y_aug, np.ones(1000) * 2)
y_aug = np.append(y_aug, np.ones(1000) * 3)

# # -- GAN-based augmentation --
# x = x_aug
# y = y_aug
# # -- GAN-based augmentation -- Comment it out to remove augmentation.

x_train, y_train = x, y
x_test, y_test = np.load('x_val.npy'), np.load('y_val.npy')
# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

# # -- Traditional augmentation --
# datagen = ImageDataGenerator(rotation_range=40,
#                              # samplewise_center=True,
#                              # width_shift_range=0.3,
#                              # height_shift_range=0.3,
#                              shear_range=0.2,
#                              zoom_range=0.2,
#                              horizontal_flip=True,
#                              vertical_flip=True)
# datagen.fit(x_train)
#
# # -- Traditional augmentation -- Comment it out to remove augmentation.

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = Sequential([
    Flatten(),
    Dense(N_NEURONS[0], input_dim=(64*64*3), kernel_initializer=weight_init),
    BatchNormalization(),
    Activation('relu'),
    Dropout(DROPOUT, seed=SEED),
])
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT, seed=SEED))
model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[
                     # EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
                     ],
          )

# model.fit(datagen.flow(x_train, y_train, batch_size=32),
#           steps_per_epoch=len(x_train) / 32,
#           epochs=N_EPOCHS,
#           validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
x_val, y_val = np.load('x_val.npy'), np.load('y_val.npy')
x_val = x_val/255.0
y_val = to_categorical(y_val, num_classes=4)
print("Final accuracy on validations set:", 100*model.evaluate(x_val, y_val)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_val), axis=1), np.argmax(y_val, axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_val), axis=1), np.argmax(y_val, axis=1), average='macro'))
y_pred = np.argmax(model.predict(x_val), axis=1)
print("Confusion Matrix: \n", confusion_matrix(np.argmax(y_val, axis=1), y_pred))

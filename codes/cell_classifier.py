# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import confusion_matrix


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 128
DROPOUT = 0.3

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


x_train, y_train = x, y
# # -- GAN-based augmentation --
# x_train, y_train = x_aug, y_aug
# # -- GAN-based augmentation -- Comment it out to remove augmentation.

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

img_size = x_train[0].shape

# %% -------------------------------------- Training Prep ----------------------------------------------------------
img = Input(img_size)
x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(img)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)
x = Dropout(DROPOUT)(x)
out = Dense(4, activation='sigmoid')(x)

model = Model(inputs=img, outputs=out)
model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])


# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[
              ModelCheckpoint("cell_classifier01.hdf5", monitor="val_loss", save_best_only=True),
              EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')
              ],
          )
# model.fit(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))
y_pred = np.argmax(model.predict(x_test), axis=1)
print("Confusion Matrix: \n", confusion_matrix(np.argmax(y_test, axis=1), y_pred))

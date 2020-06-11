# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, Flatten, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
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
BATCH_SIZE = 512
DROPOUT = 0.5

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

# -- With augmentation --
x = x_aug
y = y_aug
# -- With augmentation --

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train/255.0, x_test/255.0
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

img_size = x_train[0].shape

# %% -------------------------------------- Training Prep ----------------------------------------------------------
img = Input(img_size)
x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(img)
# x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
# x = BatchNormalization()(x)
x = MaxPooling2D(2)(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
out = Dense(4, activation='sigmoid')(x)

model = Model(inputs=img, outputs=out)
model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])


# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test))

# %% ------------------------------------------ Final test -------------------------------------------------------------
x_val, y_val = np.load('x_val.npy'), np.load('y_val.npy')
x_val = x_val/255.0
y_val = to_categorical(y_val, num_classes=4)
print("Final accuracy on validations set:", 100*model.evaluate(x_val, y_val)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_val), axis=1), np.argmax(y_val, axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_val), axis=1), np.argmax(y_val, axis=1), average='macro'))
y_pred = np.argmax(model.predict(x_val), axis=1)
print("Confusion Matrix: \n", confusion_matrix(np.argmax(y_val, axis=1), y_pred))

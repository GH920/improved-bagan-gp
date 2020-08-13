# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout, BatchNormalization,\
    Conv2D, Flatten, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.applications import ResNet50, VGG16, imagenet_utils

from skimage.transform import resize

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

inputShape = (64, 64, 3)
# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 10
BATCH_SIZE = 128
DROPOUT = 0.5
n_classes = 4

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load('x_train.npy'), np.load('y_train.npy')

# # Augmentation by GAN-generated images
# x_aug_type1 = (np.load('gen_samples_type1.npy')*0.5+0.5)*255
# x_aug_type2 = (np.load('gen_samples_type2.npy')*0.5+0.5)*255
# x_aug_type3 = (np.load('gen_samples_type3.npy')*0.5+0.5)*255
# x_aug = np.append(x, x_aug_type1, axis=0)
# x_aug = np.append(x_aug, x_aug_type2, axis=0)
# x_aug = np.append(x_aug, x_aug_type3, axis=0)
# y_aug = np.append(y, np.ones(1000))
# y_aug = np.append(y_aug, np.ones(1000) * 2)
# y_aug = np.append(y_aug, np.ones(1000) * 3)

# %% ----------------------------------- Data Augmentation -------------------------------------------------------------
# GAN based augmentation
from tensorflow.keras.models import load_model
gen_path = 'bagan_gp_cells_v3_2_epoch100.h5'
gen = load_model(gen_path)
# total_tize = len(y)
# aug_size = total_tize
#
# for c in range(n_classes):
#     sample_size = aug_size
#     label = np.ones(sample_size) * c
#     noise = np.random.normal(0, 1, (sample_size, gen.input_shape[0][1]))
#     print('Latent dimension:', gen.input_shape[0][1])
#     gen_sample = gen.predict([noise, label])
#     gen_imgs = (gen_sample*0.5 + 0.5)*255
#     x = np.append(x, gen_imgs, axis=0)
#     y = np.append(y, label)
#     print('Augmented dataset size:', sample_size, 'Total dataset size:', len(y))

x_train, y_train = x, y
# -- GAN-based augmentation --
# x_train, y_train = x_aug, y_aug
# -- GAN-based augmentation -- Comment it out to remove augmentation.

x_test, y_test = np.load('x_val.npy'), np.load('y_val.npy')

# x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=SEED, test_size=0.5, stratify=y)
# x_train, x_test = x_train-127.5, x_test-127.5
y_train, y_test = to_categorical(y_train, num_classes=n_classes), to_categorical(y_test, num_classes=4)

# # -- Traditional augmentation --
# datagen = ImageDataGenerator(rotation_range=20,
#                              # samplewise_center=True,
#                              # width_shift_range=0.3,
#                              # height_shift_range=0.3,
#                              shear_range=0.2,
#                              zoom_range=0.2,
#                              horizontal_flip=True,
#                              vertical_flip=True)
# datagen.fit(x_train)

# -- Traditional augmentation -- Comment it out to remove augmentation.

# img_size = x_train[0].shape

# x_train = scale_images(x_train, inputShape)
# x_test = scale_images(x_test, inputShape)

from tensorflow.keras.preprocessing.image import load_img

preprocess = imagenet_utils.preprocess_input
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# %% -------------------------------------- Training Prep ----------------------------------------------------------

pretrained_model = ResNet50(include_top=False, input_shape=inputShape, weights="imagenet")
for layer in pretrained_model.layers:
    layer.trainable = False
x = pretrained_model.layers[-1].output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT)(x)
out = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=pretrained_model.input, outputs=out)

model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])


# %% -------------------------------------- Training Loop ----------------------------------------------------------
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          shuffle=True,
          # callbacks=[
          #     ModelCheckpoint("cell_classifier01.hdf5", monitor="val_loss", save_best_only=True),
          #     EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')
          #     ],
          )

# model.fit(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=N_EPOCHS,
#           shuffle=True, validation_data=(x_test, y_test))

# train_loss = []
# val_loss = []
# def train(x_train, y_train, step_per_epoch=50, batch_size=128):
#     # load data
#     # n_classes = len(np.unique(y_train))
#     bs_real = batch_size
#     bs_generated = batch_size
#
#     for step in range(step_per_epoch):
#
#         idx = np.random.randint(0, x_train.shape[0], bs_real)
#         real_img = x_train[idx]
#         real_class = y_train[idx]
#
#         # Generate a batch of fake images
#         noise = np.random.normal(0, 1, (bs_generated, gen.input_shape[0][1]))
#         random_c = np.random.randint(0, n_classes, bs_generated)
#         gen_label = to_categorical(random_c, num_classes=n_classes)
#         gen_sample = gen.predict([noise, random_c])
#         gen_imgs = (gen_sample * 0.5 + 0.5) * 255
#         gen_imgs = preprocess(gen_imgs)
#
#         # Train
#         loss_gen = model.train_on_batch(gen_imgs, gen_label)
#         loss_real = model.train_on_batch(real_img, real_class)
#         train_loss.append(np.divide(np.add(loss_gen, loss_real), 2))
#
#
#         if (step + 1) * 5 % step_per_epoch == 0:
#             print(
#                 'Epoch (%d / %d): [train_loss: %f, train_acc: %.2f%%]' %
#                 (step + 1, step_per_epoch,
#                  train_loss[-1][0], 100 * train_loss[-1][-1]))
#         if step == step_per_epoch - 1:
#             loss_test = model.evaluate(x_test, y_test, verbose=0)
#             val_loss.append(loss_test)
#             print('[val_loss: %f, val_acc: %.2f%%]' % (val_loss[-1][0], 100 * val_loss[-1][-1]))
#             print("F1 score",
#                   f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))
#
#     return
#
# LEARNING_STEPS = 50
# for learning_step in range(LEARNING_STEPS):
#     print('LEARNING STEP # ', learning_step + 1, '-' * 50)
#     train(x_train, y_train, step_per_epoch=len(y_train)//BATCH_SIZE, batch_size=BATCH_SIZE)

# %% ------------------------------------------ Final test -------------------------------------------------------------

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))
y_pred = np.argmax(model.predict(x_test), axis=1)
print("Confusion Matrix: \n", confusion_matrix(np.argmax(y_test, axis=1), y_pred))

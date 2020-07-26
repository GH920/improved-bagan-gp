import os
import random
import cv2
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D, Flatten, BatchNormalization
from keras.initializers import glorot_uniform, glorot_normal
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

LR = 0.0002


real_100 = np.load('x_train.npy')
labels = np.load('y_train.npy')
real_100 = real_100[labels==1]

real = np.ndarray(shape=(real_100.shape[0], 64, 64, 3))
for i in range(real_100.shape[0]):
    real[i] = cv2.resize(real_100[i], (64, 64))

img_size = real[0].shape

# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)
trainRatio = 5

# Build Generator
def generator_conv():
    noise = Input(shape=z)
    x = Dense(4*4*128)(noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 128))(x)
    ## Out size: 4 x 4 x 128
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same')(x)
    ## Size: 8 x 8 x 128
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 16 x 16 x 128

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 32 x 32 x 128

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 64 x 64 x 128

    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    generated = Conv2D(3, (8, 8), padding='same', activation='tanh')(x)
    ## Size: 64 x 64 x 3

    generator = Model(inputs=noise, outputs=generated)
    return generator

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# clip model weights to a given hypercube
class ClipConstraint():
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# Build Discriminator
def discriminator_conv():

    const = ClipConstraint(0.01)

    img = Input(img_size)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_constraint=const)(img)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_constraint=const)(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_constraint=const)(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_constraint=const)(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_constraint=const)(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    out = Dense(1)(x)
    # out = Dense(n_classes + 1, activation='softmax')(x)

    opt = RMSprop(lr=0.00005)
    model = Model(inputs=img, outputs=out)
    model.compile(loss=wasserstein_loss, optimizer=opt)

    return model


def generator_trainer(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=RMSprop(lr=0.00005), loss=wasserstein_loss)

    return model

# GAN model compiling
class GAN():
    def __init__(self, model='conv', img_shape=(64, 64, 3), latent_space=(100,)):
        self.img_size = img_shape  # channel_last
        self.z = latent_space
        self.optimizer = Adam(0.0002, 0.5)

        if model == 'conv':
            self.gen = generator_conv()
            self.discr = discriminator_conv()
        else:
            self.gen = generator_fc()
            self.discr = discriminator_fc()

        self.train_gen = generator_trainer(self.gen, self.discr)
        self.loss_D, self.loss_G = [], []

    def train(self, imgs, epochs=50, batch_size=128):
        # load data
        imgs = (imgs - 127.5)/127.5
        bs_half = batch_size//2

        for epoch in range(epochs):
            # Get a half batch of random real images
            idx = np.random.randint(0, imgs.shape[0], bs_half)
            real_img = imgs[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, size=((bs_half,) + self.z))
            fake_img = self.gen.predict(noise)

            # fake_label = np.random.uniform(0, 0.1, (bs_half, 1))
            ## One-sided label smoothing
            real_label = np.random.uniform(0.9, 1.0, (bs_half, 1))
            fake_label = np.zeros((bs_half, 1))
            ## Random flip 5% labels/data
            mixpoint = int(bs_half * 0.95)
            real_label_mix = np.concatenate([real_label[:mixpoint], fake_label[mixpoint:]])
            fake_label_mix = np.concatenate([fake_label[:mixpoint], real_label[mixpoint:]])
            np.random.shuffle(real_label_mix)
            np.random.shuffle(fake_label_mix)
            # Train the discriminator
            loss_fake = self.discr.train_on_batch(fake_img, fake_label_mix)
            loss_real = self.discr.train_on_batch(real_img, real_label_mix)
            self.loss_D.append(0.5 * np.add(loss_fake, loss_real))

            # Train the generator
            noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
            loss_gen = self.train_gen.train_on_batch(noise, np.ones(batch_size))
            self.loss_G.append(loss_gen)

            if (epoch + 1) * 10 % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc_of_fake: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, epochs, loss_real[0], loss_fake[0], 200*self.loss_D[-1][1], loss_gen))

        return

    def train_WGAN(self, x_train, epochs=50, batch_size=128):

        # n_classes = len(np.unique(y_train))
        # bs_fake = batch_size // (n_classes + 1)
        # bs_real = batch_size - bs_fake
        bs_fake = batch_size//2
        bs_real = batch_size - bs_fake

        # lists for keeping track of loss
        d1_hist, d2_hist, g_hist = [], [], []
        for epoch in range(epochs):
            d1_tmp, d2_tmp = [], []
            for _ in range(trainRatio):
                # Get a batch of random real images
                idx = np.random.randint(0, x_train.shape[0], bs_real)
                r_img = x_train[idx]
                r_y = np.ones(bs_real)

                # Generate a batch of fake images
                noise = np.random.normal(0, 1, size=((bs_fake,) + self.z))
                f_img = self.gen.predict(noise)
                f_y = np.ones(bs_fake) * (-1)


                # train the discriminator
                # epsilon = np.random.uniform(size=(batch_size, 1, 1, 1))
                # r_loss, f_loss, penalty, d_loss = self.train_D_WGAN([r_img, latent_gen, epsilon])
                d_loss1 = self.discr.train_on_batch(r_img, r_y)
                d_loss2 = self.discr.train_on_batch(f_img, f_y)
                d1_tmp.append(d_loss1)
                d2_tmp.append(d_loss2)

            d1_hist.append(np.mean(d1_tmp))
            d2_hist.append(np.mean(d2_tmp))

            # train generator
            # Generate a batch of fake images
            noise = np.random.normal(0, 1, size=((batch_size,) + self.z))
            # g_loss = self.train_G_WGAN([latent_gen])
            f_y = np.ones(batch_size)
            g_loss = self.train_gen.train_on_batch(noise, f_y)
            g_hist.append(g_loss)

            # summarize loss on this batch
            if (epoch + 1) * 10 % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f] [Loss_G: %f]' %
                  (epoch+1, epochs, d1_hist[-1], d2_hist[-1], g_loss))
            #### Record of learning progress
            # # loss
            # self.r_loss_list.append(r_loss)
            # self.f_loss_list.append(f_loss)
            # self.f_r_loss_list.append(f_loss - r_loss)
            # self.penalty_list.append(penalty)
            # self.d_loss_list.append(d_loss)
            # self.g_loss_list.append(g_loss)

        return

def plt_img(gan, r=2, c=4):
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = gan.gen.predict(noise)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
    return

# train GAN
gan = GAN(model='conv')
LEARNING_STEPS = 100
BATCH_SIZE = 128
# EPOCHS = real.shape[0]//BATCH_SIZE
EPOCHS = 50
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    # iteration = [5, 5]
    # Adjust the learning times to balance G and D at a competitive level.
    # if gan.loss_D is not None:
    #     acc = gan.loss_D[1]
    #     iteration = [int(5 * (1-acc)) + 1, int(5 * acc) + 1]
    gan.train_WGAN(real, epochs=EPOCHS, batch_size=BATCH_SIZE)
    if (learning_step+1)%1 == 0:
        plt_img(gan)
    if (learning_step+1)%30 == 0:
        gan.gen.save('wgan_generator_%d_type1.h5' % (learning_step*100+100))

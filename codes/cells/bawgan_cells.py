import os
import random
import cv2
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Lambda
from keras.initializers import glorot_uniform, glorot_normal
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)

LR = 0.0002


# from keras.datasets.fashion_mnist import load_data
# real_100 = np.load('x_train.npy')
# (real_100, labels), (_,_) = load_data()
# X = real_100.reshape((-1, 28, 28, 1))
# # convert from ints to floats
# real_100 = X.astype('float32')
#
# real_100 = np.vstack([real_100[labels!=1], real_100[labels==1][:100]])
# labels = np.append(labels[labels!=1], np.ones(100))

real_100 = np.load('x_train.npy')
labels = np.load('y_train.npy')
# real_100 = real_100[labels==1]

channel = 3

real = np.ndarray(shape=(real_100.shape[0], 64, 64, channel))
for i in range(real_100.shape[0]):
    real[i] = cv2.resize(real_100[i], (64, 64)).reshape((64, 64, channel))

img_size = real[0].shape


x_test = np.load('x_val.npy')
y_test = np.load('y_val.npy')
x_train = real
y_train = labels

# x_train, x_test, y_train, y_test = train_test_split(real, labels, test_size=0.3, shuffle=True, random_state=42)
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5


# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)
# optimizer = RMSprop(lr=0.00005)
latent_dim=(32,)
n_classes = len(np.unique(y_train))
trainRatio = 5


# Build Generator
def generator():
    noise = Input(latent_dim)
    x = Dense(4*4*128)(noise)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 128))(x)
    ## Out size: 4 x 4 x 128
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same')(x)
    ## Size: 8 x 8 x 128
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)


    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 16 x 16 x 128
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 32 x 32 x 128
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 64 x 64 x 128
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    generated = Conv2D(channel, (8, 8), padding='same', activation='tanh')(x)
    ## Size: 64 x 64 x 3

    generator = Model(inputs=noise, outputs=generated)
    return generator

# Created by BAGAN (IBM)
def encoder(min_latent_res=8):
    # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN

    cnn = Sequential()

    cnn.add(Conv2D(16, (3, 3), padding='same', strides=(2, 2),
                   input_shape=img_size, use_bias=True))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), use_bias=True))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, (3, 3), padding='same', strides=(2, 2), use_bias=True))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), use_bias=True))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    while cnn.output_shape[-2] > min_latent_res:
        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

    cnn.add(Flatten())
    cnn.add(Dense(latent_dim[0], activation='relu'))

    img_input = Input(img_size)
    features_output = cnn(img_input)
    model = Model(img_input, features_output)
    # model2 = Sequential()
    # model2.add(model)
    # model2.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))  # ???axis=0?1

    return model

def generator2():
    encoded = Input(latent_dim)
    x = Dense(4 * 4 * 8)(encoded)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 8))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(channel, (3, 3), activation='tanh', padding='same')(x)

    model = Model(encoded, decoded)

    return model

def encoder2():
    img_input = Input(img_size)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim[0], activation='relu')(x)

    model = Model(img_input, encoded)
    return model


def autoencoder_trainer(encoder, decoder):

    model = Sequential()
    model.add(encoder)
    # model.add(Lambda(lambda x: K.l2_normalize(x, axis=0))) #???axis=0?1
    model.add(decoder)

    model.compile(optimizer=optimizer, loss='mae')
    return model

en = load_model('encoder_cells_epoch50.h5')
de = load_model('decoder_cells_epoch50.h5')
en.name = 'encoder'
de.name = 'decoder'
# ae = autoencoder_trainer(en, de)
#
# ae.fit(x_train, x_train,
#        epochs=50,
#        batch_size=128,
#        shuffle=True,
#        validation_data=(x_test, x_test))

# decoded_imgs = ae.predict(x_test)
# n = n_classes
# plt.figure(figsize=(2*n, 4))
# decoded_imgs = decoded_imgs*0.5 + 0.5
# x_real = x_test*0.5 + 0.5
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i+1)
#     if channel == 3:
#         plt.imshow(x_real[y_test==i][0].reshape(64, 64, channel))
#     else:
#         plt.imshow(x_real[y_test==i][0].reshape(64, 64))
#         plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # display reconstruction
#     ax = plt.subplot(2, n, i + n + 1)
#     if channel == 3:
#         plt.imshow(decoded_imgs[y_test==i][0].reshape(64, 64, channel))
#     else:
#         plt.imshow(decoded_imgs[y_test==i][0].reshape(64, 64))
#         plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

means, covs = None, None
for i in range(n_classes):
    f = en.predict(x_train[y_train == i])
    if i == 0:
        means = np.mean(f, axis=0)
        covs = np.array([np.cov(f.T)])
    else:
        means = np.vstack([means, np.mean(f, axis=0)])
        covs = np.vstack([covs, np.array([np.cov(f.T)])])



def generate_latent(c, means, covs):  # c is a vector of classes
    latent = np.array([
        np.random.multivariate_normal(means[e], covs[e])
        for e in c
    ])
    return latent


# Build Discriminator
def discriminator(encoder):
    # img = Input(img_size)
    # x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)
    # x = LeakyReLU(0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Flatten()(x)
    # x = Dropout(0.4)(x)

    # model = Model(inputs=img, outputs=out)

    model = Sequential()
    model.add(encoder)
    model.add(Dense(n_classes + 1, activation='softmax'))

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def discriminator_wgan():

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

# def generator_trainer(generator, discriminator):
#
#     discriminator.trainable = False
#
#     model = Sequential()
#     model.add(generator)
#     model.add(discriminator)
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
#
#     return model

def generator_trainer(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=RMSprop(lr=0.00005), loss=wasserstein_loss)

    return model

# gradient penalty coefficient "λ"
penaltyLambda = 10    # d_loss = f_loss - r_loss + λ･penalty

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


# def build_WGANgp(generator, discriminator):
#     #### model
#     # generator image(fake image)
#     z = Input(shape=latent_dim)
#     f_img = generator(z)
#     f_out = discriminator(f_img)
#     # real image
#     r_img = Input(shape=img_size)
#     r_out = discriminator(r_img)
#     # average image
#     epsilon = K.placeholder(shape=(None, 1, 1, 1))
#     a_img = Input(shape=(img_size),
#                   tensor=epsilon * r_img + (1 - epsilon) * f_img)
#     a_out = discriminator(a_img)
#
#     #### loss
#     # original critic(discriminator) loss
#     r_loss = K.mean(r_out)
#     f_loss = K.mean(f_out)
#     # gradient penalty  <this is point of WGAN-gp>
#     grad_mixed = K.gradients(a_out, [a_img])[0]
#     norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
#     grad_penalty = K.mean(K.square(norm_grad_mixed - 1))
#     penalty = penaltyLambda * grad_penalty
#     # d loss
#     d_loss = f_loss - r_loss + penalty
#
#     #### discriminator update function
#     d_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9). \
#         get_updates(params=discriminator.trainable_weights, loss=d_loss)
#     d_train = K.function(inputs=[r_img, z, epsilon],
#                          outputs=[r_loss, f_loss, penalty, d_loss],
#                          updates=d_updates)
#
#     #### generator update function
#     g_loss = -1. * f_loss
#     g_updates = Adam(lr=1e-4, beta_1=0.5, beta_2=0.9). \
#         get_updates(params=generator.trainable_weights, loss=g_loss)
#     g_train = K.function(inputs=[z], outputs=[g_loss], updates=g_updates)
#
#     return g_train, d_train

# GAN model compiling
class BAGAN():
    def __init__(self, img_shape=(64, 64, channel)):
        self.img_size = img_shape  # channel_last
        self.optimizer = Adam(0.0002, 0.5)

        self.encoder = encoder()
        self.gen = de
        self.discr = discriminator_wgan()

        self.train_ae = autoencoder_trainer(self.encoder, self.gen)

        # GAN Trainer
        self.train_gen = generator_trainer(self.gen, self.discr)

        # WGAN Trainer
        # self.train_G_WGAN, self.train_D_WGAN = build_WGANgp(self.gen, self.discr)

        self.loss_D, self.loss_G = [], []

        # list for store learning progress data
        self.g_loss_list = []
        self.r_loss_list = []
        self.f_loss_list = []
        self.f_r_loss_list = []
        self.penalty_list = []
        self.d_loss_list = []

    def train_autoencoder(self, x_train, x_test, epochs=5, batch_size=128):
        self.train_ae.fit(x_train, x_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(x_test, x_test))
        return

    def train(self, x_train, y_train, epochs=50, batch_size=128):
        # load data
        n_classes = len(np.unique(y_train))
        bs_fake = batch_size // (n_classes + 1)
        bs_real_each_c = (batch_size - bs_fake) // n_classes
        bs_real = (batch_size - bs_fake)

        for epoch in range(epochs):
            # Get a batch of random real images
            real_img, real_class = None, None
            for c in range(n_classes):
                idx = np.random.randint(0, x_train[y_train==c].shape[0], bs_real_each_c)
                if c == 0:
                    real_img = x_train[y_train==c][idx]
                    real_class = np.ones(bs_real_each_c) * c
                else:
                    real_img = np.vstack([real_img, x_train[y_train==c][idx]])
                    real_class = np.append(real_class, np.ones(bs_real_each_c) * c)

            real_img, real_class = shuffle(real_img, real_class)

            # idx = np.random.randint(0, x_train.shape[0], bs_real)
            # real_img = x_train[idx]
            # real_class = y_train[idx]

            # Generate a batch of fake images
            random_c = np.random.randint(0, n_classes, bs_fake)
            latent_gen = generate_latent(random_c, means, covs)
            fake_img = self.gen.predict(latent_gen)

            for _ in range(trainRatio):
                # Train the discriminator
                loss_fake = self.discr.train_on_batch(fake_img, np.ones(bs_fake)*n_classes)
                loss_real = self.discr.train_on_batch(real_img, real_class)
            self.loss_D.append(np.add(loss_fake, loss_real))

            # Train the generator
            random_c = np.random.randint(0, n_classes, batch_size)
            latent_gen = generate_latent(random_c, means, covs)
            loss_gen = self.train_gen.train_on_batch(latent_gen, random_c)
            self.loss_G.append(loss_gen)

            if (epoch + 1) * 10 % epochs == 0:
                print('Epoch (%d / %d): [Loss_D_real: %f, Loss_D_fake: %f, acc_of_fake: %.2f%%, acc_of_real: %.2f%%] [Loss_G: %f]' %
                  (epoch+1, epochs, loss_real[0], loss_fake[0], 100*loss_fake[-1], 100*loss_real[-1], loss_gen))

        return

    def train_WGAN(self, x_train, y_train, epochs=50, batch_size=128):

        n_classes = len(np.unique(y_train))
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
                random_c = np.random.randint(0, n_classes, bs_fake)
                latent_gen = generate_latent(random_c, means, covs)
                f_img = self.gen.predict(latent_gen)
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
            random_c = np.random.randint(0, n_classes, batch_size)
            latent_gen = generate_latent(random_c, means, covs)
            # g_loss = self.train_G_WGAN([latent_gen])
            f_y = np.ones(batch_size)
            g_loss = self.train_gen.train_on_batch(latent_gen, f_y)
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


def plt_img(gan):
    latent_gen = generate_latent(list(range(n_classes)), means, covs)
    decoded_imgs = gan.gen.predict(latent_gen)
    decoded_imgs = decoded_imgs * 0.5 + 0.5
    x_real = x_test * 0.5 + 0.5
    n = n_classes
    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test==i][0].reshape(64, 64, channel))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(64, 64))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display generation
        ax = plt.subplot(2, n, i + n + 1)
        if channel == 3:
            plt.imshow(decoded_imgs[i].reshape(64, 64, channel))
        else:
            plt.imshow(decoded_imgs[i].reshape(64, 64))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return

# # train GAN
bagan = BAGAN()
# BATCH_SIZE = 64
# # EPOCHS = real.shape[0]//BATCH_SIZE
# EPOCHS = 100
# train GAN
LEARNING_STEPS = 10
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step+1, '-'*50)
    # iteration = [5, 5]
    # Adjust the learning times to balance G and D at a competitive level.
    # if gan.loss_D is not None:
    #     acc = gan.loss_D[1]
    #     iteration = [int(5 * (1-acc)) + 1, int(5 * acc) + 1]
    bagan.train_WGAN(x_train, y_train)
    if (learning_step+1)%1 == 0:
        plt_img(bagan)
    # if (learning_step+1)%50 == 0:
    #     gan.gen.save('gan_generator_%d_type1.h5' % (learning_step*100+100))

# bawgan = BAGAN()
# # generated image sumple
# LEARNING_STEPS = 31
# for learning_step in range(LEARNING_STEPS):
#     bawgan.train_WGAN(x_train, y_train)
#     if (learning_step+1)%1 == 0:
#         plt_img(bawgan)

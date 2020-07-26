import os
import random
import cv2
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, \
    MaxPooling2D, UpSampling2D, Flatten, BatchNormalization, Lambda
from tensorflow.keras.initializers import glorot_uniform, glorot_normal
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


# from tensorflow.keras.datasets.fashion_mnist import load_data
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

x_train, x_test, y_train, y_test = train_test_split(real, labels, test_size=0.3, shuffle=True, random_state=42)
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_test = (x_test.astype('float32') - 127.5) / 127.5


# latent space of noise
z = (100,)
optimizer = Adam(lr=0.0002, beta_1=0.5)
latent_dim=(32,)
n_classes = len(np.unique(y_train))
trainRatio = 5
# means = np.load('means_epoch50.npy')
# covs = np.load('covs_epoch50.npy')

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
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)


    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 16 x 16 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 32 x 32 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    ## Size: 64 x 64 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    generated = Conv2D(channel, (2, 2), padding='same', activation='tanh')(x)
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
    cnn.add(Dense(latent_dim[0], activation='tanh'))

    img_input = Input(img_size)
    features_output = cnn(img_input)
    model = Model(img_input, features_output)
    # model2 = Sequential()
    # model2.add(model)
    # model2.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))  # ???axis=0?1

    return model

def autoencoder_trainer(encoder, decoder):

    model = Sequential()
    model.add(encoder)
    # model.add(Lambda(lambda x: K.l2_normalize(x, axis=0))) #???axis=0?1
    model.add(decoder)

    model.compile(optimizer=optimizer, loss='mae')
    return model

en = encoder()
de = generator()
ae = autoencoder_trainer(en, de)

ae.fit(x_train, x_train,
       epochs=3,
       batch_size=128,
       shuffle=True,
       validation_data=(x_test, x_test))

decoded_imgs = ae.predict(x_test)
n = n_classes
plt.figure(figsize=(2*n, 4))
decoded_imgs = decoded_imgs*0.5 + 0.5
x_real = x_test*0.5 + 0.5
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    if channel == 3:
        plt.imshow(x_real[y_test==i][0].reshape(64, 64, channel))
    else:
        plt.imshow(x_real[y_test==i][0].reshape(64, 64))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    if channel == 3:
        plt.imshow(decoded_imgs[y_test==i][0].reshape(64, 64, channel))
    else:
        plt.imshow(decoded_imgs[y_test==i][0].reshape(64, 64))
        plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

means, covs = None, None
for i in range(n_classes):
    f = en.predict(x_train[y_train == i])
    if i == 0:
        means = np.mean(f, axis=0)
        covs = np.array([np.cov(f.T)])
    else:
        means = np.vstack([means, np.mean(f, axis=0)])
        covs = np.vstack([covs, np.array([np.cov(f.T)])])


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

    img = Input(img_size)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)
    x = Flatten()(x)
    out = Dense(1)(x)
    # out = Dense(n_classes + 1, activation='softmax')(x)

    # opt = RMSprop(lr=0.00005)
    model = Model(inputs=img, outputs=out)
    # model.compile(loss=wasserstein_loss, optimizer=opt)

    return model

def generator_trainer(generator, discriminator):

    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    return model

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
        self.train_G_WGAN, self.train_D_WGAN = build_WGANgp(self.gen, self.discr)

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

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discr(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp


    def train_WGAN(self, x_train, y_train, epochs=50, batch_size=128):

        ########check this part code!!!

        n_classes = len(np.unique(y_train))
        # bs_fake = batch_size // (n_classes + 1)
        # bs_real = batch_size - bs_fake
        bs_fake = batch_size//2
        bs_real = batch_size - bs_fake

        # lists for keeping track of loss
        for epoch in range(epochs):
            # Get a batch of random real images
            idx = np.random.randint(0, x_train.shape[0], bs_real)
            real_images = x_train[idx]
            r_y = np.ones(bs_real)

            # Generate a batch of fake images
            random_c = np.random.randint(0, n_classes, bs_fake)
            latent_gen = generate_latent(random_c, means, covs)
            f_img = self.gen.predict(latent_gen)
            f_y = np.ones(bs_fake) * (-1)
            for _ in range(trainRatio):
                with tf.GradientTape() as tape:
                    # Generate fake images from the latent vector
                    fake_images = self.gen(random_latent_vectors, training=True)
                    # Get the logits for the fake images
                    fake_logits = self.discr(fake_images, training=True)
                    # Get the logits for real images
                    real_logits = self.discr(real_images, training=True)

                    # Calculate discriminator loss using fake and real logits
                    d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                    # Calculate the gradient penalty
                    gp = self.gradient_penalty(batch_size, real_images, fake_images)
                    # Add the gradient penalty to the original discriminator loss
                    d_loss = d_cost + gp * self.gp_weight





                # train the discriminator
                epsilon = np.random.uniform(size=(batch_size, 1, 1, 1))
                r_loss, f_loss, penalty, d_loss = self.train_D_WGAN([r_img, latent_gen, epsilon])

            # train generator
            # Generate a batch of fake images
            random_c = np.random.randint(0, n_classes, batch_size*2)
            latent_gen = generate_latent(random_c, means, covs)
            g_loss = self.train_G_WGAN([latent_gen])

            #### Record of learning progress
            # loss
            self.r_loss_list.append(r_loss)
            self.f_loss_list.append(f_loss)
            self.f_r_loss_list.append(f_loss - r_loss)
            self.penalty_list.append(penalty)
            self.d_loss_list.append(d_loss)
            self.g_loss_list.append(g_loss)

        return

# def generate_latent_wgan(c, means, covs):  # c is a vector of classes
#     latent = tf.convert_to_tensor([
#         tfp.distributions.MultivariateNormalFullCovariance(means[e], covs[e])
#         for e in c
#     ])
#     return latent

# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# means = np.divide(means, np.max(means, axis=0))
# covs = np.divide(covs, np.max(covs, axis=0)) # ???covs matrix may not need to be normalized

covs = covs.astype('float32')

means = tf.convert_to_tensor(means, dtype=tf.float32)
covs = tf.convert_to_tensor(covs, dtype=tf.float32)

def generate_latent(c, means, covs):  # c is a vector of classes
    latent = []
    for e in c:
        latent.append(np.random.multivariate_normal(means[e], covs[e]))
        # np.random.multivariate_normal() will automatically turn float32 into float64
    return np.array(latent).astype('float32') # float32 is IMPORTANT for tf graph

class WGAN(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[n_classes, latent_dim[0]], dtype=tf.float32),
        tf.TensorSpec(shape=[n_classes, latent_dim[0], latent_dim[0]], dtype=tf.float32)
    ])
    def tf_generate_latent(self, c, means, covs):  # c is a vector of classes
        latent = tf.numpy_function(generate_latent, [c, means, covs], tf.float32)
        return latent

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add gradient penalty to the discriminator loss
        # 6. Return generator and discriminator losses as a loss dictionary.

        # Train discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_c = tf.random.uniform([batch_size], 0, n_classes, dtype=tf.int32)

            random_latent_vectors = self.tf_generate_latent(random_c, means, covs)
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        random_c = tf.random.uniform([batch_size], 0, n_classes, dtype=tf.int32)
        random_latent_vectors = self.tf_generate_latent(random_c, means, covs)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

# Optimizer for both the networks
# learning_rate=0.0002, beta_1=0.5 are recommended
# optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions to be used for discrimiator
# This should be (fake_loss - real_loss)
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

# Epochs to train
epochs = 20

from tensorflow.keras.models import load_model
# Get the wgan model
d_model = discriminator_wgan()
# g_model = load_model('decoder_cells_epoch50.h5')
g_model = de

wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim[0],
    discriminator_extra_steps=3,
)

# Compile the wgan model
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

def plt_img(wgan):
    # latent_gen = generate_latent(list(range(n_classes)), means, covs)
    latent_gen = np.random.normal(size=(10, latent_dim[0]))
    decoded_imgs = wgan.generator.predict(latent_gen)
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

# Start training
# wgan.fit(x_train, batch_size=128, epochs=epochs)

LEARNING_STEPS = 50
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 50)
    wgan.fit(x_train, batch_size=128, epochs=1)
    if (learning_step+1)%1 == 0:
        plt_img(wgan)

# def plt_img(gan):
#     latent_gen = generate_latent(list(range(n_classes)), means, covs)
#     decoded_imgs = gan.gen.predict(latent_gen)
#     decoded_imgs = decoded_imgs * 0.5 + 0.5
#     x_real = x_test * 0.5 + 0.5
#     n = n_classes
#     plt.figure(figsize=(2*n, 4))
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         if channel == 3:
#             plt.imshow(x_real[y_test==i][0].reshape(64, 64, channel))
#         else:
#             plt.imshow(x_real[y_test == i][0].reshape(64, 64))
#             plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         # display generation
#         ax = plt.subplot(2, n, i + n + 1)
#         if channel == 3:
#             plt.imshow(decoded_imgs[i].reshape(64, 64, channel))
#         else:
#             plt.imshow(decoded_imgs[i].reshape(64, 64))
#             plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()
#     return




# # # train GAN
# bagan = BAGAN()
# # BATCH_SIZE = 64
# # # EPOCHS = real.shape[0]//BATCH_SIZE
# # EPOCHS = 100
# # train GAN
# LEARNING_STEPS = 10
# for learning_step in range(LEARNING_STEPS):
#     print('LEARNING STEP # ', learning_step+1, '-'*50)
#     # iteration = [5, 5]
#     # Adjust the learning times to balance G and D at a competitive level.
#     # if gan.loss_D is not None:
#     #     acc = gan.loss_D[1]
#     #     iteration = [int(5 * (1-acc)) + 1, int(5 * acc) + 1]
#     bagan.train_WGAN(x_train, y_train)
#     if (learning_step+1)%1 == 0:
#         plt_img(bagan)
#     # if (learning_step+1)%50 == 0:
#     #     gan.gen.save('gan_generator_%d_type1.h5' % (learning_step*100+100))
#


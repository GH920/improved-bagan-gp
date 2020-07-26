import os
import random
import cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, Concatenate, multiply, \
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


# Build Generator
def generator():
    # label = Input((1,), dtype='int32')
    noise = Input(latent_dim)
    #
    # le = Flatten()(Embedding(n_classes, latent_dim[0])(label))
    # noise_le = multiply([noise, le])

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
    cnn.add(Dense(latent_dim[0], activation='relu'))

    img_input = Input(img_size)
    features_output = cnn(img_input)
    model = Model(img_input, features_output)
    # model2 = Sequential()
    # model2.add(model)
    # model2.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))  # ???axis=0?1

    return model

def embedding_labeled_latent():
    label = Input((1,), dtype='int32')
    noise = Input(latent_dim)

    le = Flatten()(Embedding(n_classes, latent_dim[0])(label))
    noise_le = multiply([noise, le])

    model = Model([noise, label], noise_le)

    return model


def autoencoder_trainer(encoder, decoder, embedding):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    latent = encoder(img)
    labeled_latent = embedding_labeled_latent([latent, label])
    rec_img = decoder(labeled_latent)
    model = Model([img, label], rec_img)

    # model = Sequential()
    # model.add(encoder)
    # model.add(embedding)
    # model.add(decoder)

    model.compile(optimizer=optimizer, loss='mae')
    return model

en = encoder()
de = generator()
em = embedding_labeled_latent()
ae = autoencoder_trainer(en, de, em)

ae.fit([x_train, y_train], x_train,
       epochs=3,
       batch_size=128,
       shuffle=True,
       validation_data=([x_test, y_test], x_test))

decoded_imgs = ae.predict([x_test, y_test])
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
#
# means, covs = None, None
# for i in range(n_classes):
#     f = en.predict(x_train[y_train == i])
#     if i == 0:
#         means = np.mean(f, axis=0)
#         covs = np.array([np.cov(f.T)])
#     else:
#         means = np.vstack([means, np.mean(f, axis=0)])
#         covs = np.vstack([covs, np.array([np.cov(f.T)])])



# def generate_latent(c, means, covs):  # c is a vector of classes
#     latent = np.array([
#         np.random.multivariate_normal(means[e], covs[e])
#         for e in c
#     ])
#     return latent


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
    label = Input((1,), dtype='int32')

    le = Flatten()(Embedding(n_classes, np.prod(img_size))(label))
    flat_img = Flatten()(img)
    img_le = multiply([flat_img, le])

    x = Dense(np.prod(img_size))(img_le)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape(img_size)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    out = Dense(1)(x)
    # class_out = Dense(n_classes, activation='softmax')(x)


    # opt = RMSprop(lr=0.00005)
    model = Model(inputs=[img, label], outputs=out)
    # model.compile(loss=wasserstein_loss, optimizer=opt)

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

    def gradient_penalty(self, batch_size, real_images, fake_images, labels):
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
            pred = self.discriminator([interpolated, labels], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            real_images = data[0]
            labels = data[1]

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
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, labels], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([real_images, labels], training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits,
                                        # pred_real_class=class_real, pred_fake_class=class_fake,
                                        # real_label=labels
                                        )

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, labels)
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
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, labels], training=True)
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
def discriminator_loss(real_logits, fake_logits,
                       # pred_real_class, pred_fake_class, real_label
                       ):
    # real_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    # fake_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    # real_class_loss = tf.reduce_mean(
    #     tf.keras.losses.sparse_categorical_crossentropy(y_true=real_label, y_pred=pred_real_class))
    # fake_class_loss = tf.reduce_mean(
    #     tf.keras.losses.sparse_categorical_crossentropy(y_true=real_label, y_pred=pred_fake_class))

    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)

    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_logits,
                   # pred_fake_class, real_label
                   ):
    # fake_loss = tf.reduce_mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    # fake_class_loss = tf.reduce_mean(
    #     tf.keras.losses.sparse_categorical_crossentropy(y_true=real_label, y_pred=pred_fake_class))

    fake_loss = tf.reduce_mean(fake_logits)
    return -fake_loss

# Epochs to train
epochs = 20

def generator_label(embedding, decoder):

    model = Sequential()
    model.add(embedding)
    model.add(decoder)

    return model


from tensorflow.keras.models import load_model
# Get the wgan model
d_model = discriminator_wgan()
# g_model = load_model('decoder_cells_epoch50.h5')
g_model = generator_label(em, de)

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
    np.random.seed(42)
    # latent_gen = generate_latent(list(range(n_classes)), means, covs)
    latent_gen = np.random.normal(size=(n_classes, latent_dim[0]))

    x_real = x_test * 0.5 + 0.5
    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test==i][0].reshape(64, 64, channel))
        else:
            plt.imshow(x_real[y_test == i][0].reshape(64, 64))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = wgan.generator.predict([latent_gen, np.ones(n)*c])
            decoded_imgs = decoded_imgs * 0.5 + 0.5
            # display generation
            ax = plt.subplot(n+1, n, (i+1)*n + 1 + c)
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
    wgan.fit(x_train, y_train, batch_size=128, epochs=2)
    if (learning_step+1)%1 == 0:
        plt_img(wgan)

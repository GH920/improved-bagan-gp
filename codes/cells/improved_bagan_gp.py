import os
import random
import cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, \
    Activation, LeakyReLU, Conv2D, Conv2DTranspose, Embedding, \
    Concatenate, multiply, Flatten, BatchNormalization
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_normal(seed=SEED)


# from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.datasets.cifar10 import load_data
(real_100, labels), (_,_) = load_data()
# X = real_100.reshape((-1, 28, 28, 1))
labels = labels.reshape(-1)
# convert from ints to floats
real_100 = real_100.astype('float32')

# # create imbalanced datasets
# for c in range(1, 10):
#     real_100 = np.vstack([real_100[labels!=c], real_100[labels==c][:100*c]])
#     labels = np.append(labels[labels!=c], np.ones(100*c) * c)

# # use our datasets
# real_100 = np.load('x_train.npy')
# labels = np.load('y_train.npy')
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
optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
latent_dim=(128,)
n_classes = len(np.unique(y_train))
trainRatio = 5

# Build Generator/Decoder
def decoder():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    noise_le = Input(latent_dim)

    x = Dense(4*4*256)(noise_le)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, 256))(x)
    ## Out size: 4 x 4 x 256
    x = Conv2DTranspose(filters=128,
                        kernel_size=(4, 4),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=init)(x)
    ## Size: 8 x 8 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)


    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    ## Size: 16 x 16 x 128
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    ## Size: 32 x 32 x 64
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    generated = Conv2DTranspose(channel, (4, 4), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=init)(x)
    ## Size: 64 x 64 x 3

    generator = Model(inputs=noise_le, outputs=generated)
    return generator

def encoder():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(img_size)
    label = Input((1,), dtype='int32')

    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(img)
    # x = LayerNormalization()(x) # It is not suggested to use BN in Discriminator of WGAN
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    # 4 x 4 x 256
    feature = Flatten()(x)

    feature = Dense(latent_dim[0])(feature)
    out = LeakyReLU(0.2)(feature)

    model = Model(inputs=img, outputs=out)
    return model

def embedding_labeled_latent():
    # # weight initialization
    # init = RandomNormal(stddev=0.02)

    label = Input((1,), dtype='int32')
    noise = Input(latent_dim)
    # ne = Dense(256)(noise)
    # ne = LeakyReLU(0.2)(ne)

    le = Flatten()(Embedding(n_classes, latent_dim[0])(label))
    # le = Dense(256)(le)
    # le = LeakyReLU(0.2)(le)

    noise_le = multiply([noise, le])
    # noise_le = Dense(latent_dim[0])(noise_le)

    model = Model([noise, label], noise_le)

    return model


def autoencoder_trainer(encoder, decoder, embedding):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    latent = encoder(img)
    labeled_latent = embedding([latent, label])
    rec_img = decoder(labeled_latent)
    model = Model([img, label], rec_img)

    model.compile(optimizer=optimizer, loss='mae')
    return model

en = encoder()
de = decoder()
em = embedding_labeled_latent()
ae = autoencoder_trainer(en, de, em)

ae.fit([x_train, y_train], x_train,
       epochs=30,
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

# from tensorflow.keras.models import load_model
# en = load_model('../bagan_gp_encoder.h5')
# em = load_model('../bagan_gp_embedding.h5')
# de = load_model('../bagan_gp_decoder.h5')

# Build Discriminator, similar to cWGAN
def discriminator_wgan():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    img = Input(img_size)
    label = Input((1,), dtype='int32')


    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(img)
    # x = LayerNormalization()(x) # It is not suggested to use BN in Discriminator of WGAN
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(x)
    # x = LayerNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = Dropout(0.3)(x)

    x = Flatten()(x)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(4 * 4 * 256)(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[img, label], outputs=out)

    return model

class BAGAN_GP(Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(BAGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(BAGAN_GP, self).compile()
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

        # For each batch, we are going to perform wgan-like process
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
            wrong_labels = tf.random.uniform((batch_size,), 0, n_classes)  # fake as a new class
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, fake_labels], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, fake_labels], training=True)
                # Get the logits for real images
                real_logits = self.discriminator([real_images, labels], training=True)
                # Get the logits for wrong label classification
                wrong_label_logits = self.discriminator([real_images, wrong_labels], training=True)

                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits,
                                        wrong_label_logits=wrong_label_logits
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
        fake_labels = tf.random.uniform((batch_size,), 0, n_classes)
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, fake_labels], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, fake_labels], training=True)
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
# learning_rate=0.0002, beta_1=0.5, beta_2=0.9 are recommended
generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)

# Define the loss functions to be used for discrimiator
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_logits, fake_logits, wrong_label_logits):
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
    wrong_label_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_label_logits, labels=tf.zeros_like(fake_logits)))

    return wrong_label_loss + fake_loss + real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_logits):
    fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
    return fake_loss

# build generator with pretrained decoder and embedding
def generator_label(embedding, decoder):

    # embedding.trainable = False

    label = Input((1,), dtype='int32')
    latent = Input(latent_dim)

    labeled_latent = embedding([latent, label])
    gen_img = decoder(labeled_latent)
    model = Model([latent, label], gen_img)

    return model

# build discriminator with pretrained encoder
def build_discriminator(encoder):

    label = Input((1,), dtype='int32')
    img = Input(img_size)

    inter_output_model = Model(inputs=encoder.input, outputs=encoder.layers[-3].output)
    x = inter_output_model(img)

    le = Flatten()(Embedding(n_classes, 512)(label))
    le = Dense(4 * 4 * 256)(le)
    le = LeakyReLU(0.2)(le)
    x_y = multiply([x, le])
    x_y = Dense(512)(x_y)

    out = Dense(1)(x_y)

    model = Model(inputs=[img, label], outputs=out)

    return model


from tensorflow.keras.models import load_model
# Get the wgan-like model
# d_model = build_discriminator(en)
d_model = discriminator_wgan()
# g_model = load_model('decoder_cells_epoch50.h5')
g_model = generator_label(em, de)

bagan_gp = BAGAN_GP(
    discriminator=d_model,
    generator=g_model,
    latent_dim=latent_dim[0],
    discriminator_extra_steps=3,
)

# Compile the model
bagan_gp.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

def plt_img(generator):
    np.random.seed(42)
    latent_gen = np.random.normal(size=(n_classes, latent_dim[0]))

    x_real = x_test * 0.5 + 0.5
    n = n_classes

    plt.figure(figsize=(2*n, 2*(n+1)))
    for i in range(n):
        # display original
        ax = plt.subplot(n+1, n, i + 1)
        if channel == 3:
            plt.imshow(x_real[y_test==i][4].reshape(64, 64, channel))
        else:
            plt.imshow(x_real[y_test == i][4].reshape(64, 64))
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for c in range(n):
            decoded_imgs = generator.predict([latent_gen, np.ones(n)*c])
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
LEARNING_STEPS = 10
for learning_step in range(LEARNING_STEPS):
    print('LEARNING STEP # ', learning_step + 1, '-' * 50)
    bagan_gp.fit(x_train, y_train, batch_size=128, epochs=2)
    if (learning_step+1)%1 == 0:
        plt_img(bagan_gp.generator)

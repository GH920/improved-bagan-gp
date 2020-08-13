import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.models import load_model

def gen_samples(n=1000, celltype=1):
    gen_path = 'gan_generator_mlp_5000_type%d.h5' % celltype
    gen = load_model(gen_path)

    sample_size = n
    noise = np.random.normal(0, 1, (sample_size, 100))
    gen_sample = gen.predict(noise)
    gen_sample = gen_sample*0.5 + 0.5

    np.save('gen_samples_mlp_type%d' % celltype, gen_sample)

# gen_samples(1000, 1)
# gen_samples(1000, 2)
# gen_samples(1000, 3)

def generate_latent(c, means, covs):  # c is a vector of classes
    latent = np.array([
        np.random.multivariate_normal(means[e], covs[e])
        for e in c
    ])
    return latent

def bagan_gen_samples(n=1000, type=1):
    # gen_path = 'bagan_generator_6000.h5'
    gen_path = 'decoder_cells_epoch50.h5'
    gen = load_model(gen_path)
    means = np.load('means_epoch50.npy')
    covs = np.load('covs_epoch50.npy')

    sample_size = n
    labels = np.ones(sample_size).astype('int') * type
    latent_gen = generate_latent(labels, means, covs)
    decoded_imgs = gen.predict(latent_gen)
    decoded_imgs = decoded_imgs * 0.5 + 0.5

    # np.save('bagan_gen_x_type%d.npy' % type, decoded_imgs)
    np.save('ae_rec_x_type%d.npy' % type, decoded_imgs)

# bagan_gen_samples(1000, 0)
# bagan_gen_samples(1000, 1)
# bagan_gen_samples(1000, 2)
# bagan_gen_samples(1000, 3)

# def autoencoder_rec_samples(n=1000, n_classes=4):
#     gen_path = 'decoder_cells_epoch50.h5'
#     gen = load_model(gen_path)
#     means = np.load('means_epoch50.npy')
#     covs = np.load('covs_epoch50.npy')
#
#     sample_size = n
#     random_c = np.random.randint(0, n_classes, sample_size)
#     latent_gen = generate_latent(random_c, means, covs)
#     decoded_imgs = gen.predict(latent_gen)
#     decoded_imgs = decoded_imgs * 0.5 + 0.5
#
#     np.save('reconstructed_samples_alltype.npy', decoded_imgs)
#
# autoencoder_rec_samples()

def bagan_gen_samples_alltype(n=1000, n_classes=4):
    gen_path = 'bagan_generator_15000.h5'
    gen = load_model(gen_path)
    means = np.load('means_epoch20_norm.npy')
    covs = np.load('covs_epoch20_norm.npy')

    sample_size = n
    random_c = np.random.randint(0, n_classes, sample_size)
    latent_gen = generate_latent(random_c, means, covs)
    decoded_imgs = gen.predict(latent_gen)
    decoded_imgs = decoded_imgs * 0.5 + 0.5

    np.save('bagan_gen_15000_x_alltype.npy', decoded_imgs)

# bagan_gen_samples_alltype()

def wgan_gen_samples(n=1000):
    gen_path = 'cwgan_gp_basic_cells.h5'
    gen = load_model(gen_path)

    sample_size = n
    noise = np.random.normal(0, 1, (sample_size, 32))
    gen_sample = gen.predict(noise)
    gen_sample = gen_sample*0.5 + 0.5

    np.save('wgan_gp_resnet_noNorm_gen_samples_alltype.npy', gen_sample)

# wgan_gen_samples(1000)

def cwgan_gen_samples(n=1000, type=0):
    gen_path = 'cwgan_gp_basic_cells.h5'
    gen = load_model(gen_path)

    sample_size = n
    label = np.ones(1000) * type
    noise = np.random.normal(0, 1, (sample_size, 32))
    gen_sample = gen.predict([noise, label])
    gen_sample = gen_sample*0.5 + 0.5

    np.save('cwgan_gp_basic_cells_gen_samples_type%d.npy' % type, gen_sample)

cwgan_gen_samples(1000, 0)
cwgan_gen_samples(1000, 1)
cwgan_gen_samples(1000, 2)
cwgan_gen_samples(1000, 3)
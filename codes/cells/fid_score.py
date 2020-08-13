import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize

# Reference: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)*255
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))



from tensorflow.keras.models import load_model
gen_path = 'cwgan_gp_modified_cells_epoch50.h5'
gen = load_model(gen_path)

n_classes = 4
for c in range(n_classes):
    # load generator
    sample_size = 1000
    label = np.ones(sample_size) * c
    noise = np.random.normal(0, 1, (sample_size, gen.input_shape[0][1]))
    print('Latent dimension:', gen.input_shape[0][1])
    gen_sample = gen.predict([noise, label])
    gen_imgs = gen_sample*0.5 + 0.5

    # # load generated/real data
    # gen_imgs = np.load('bagan_gen_x_type%d.npy' % c)
    # gen_imgs = gen_imgs * 0.5 + 0.5
    # gen_imgs = np.load('x_train.npy')
    # gen_label = np.load('y_train.npy')
    # gen_imgs = gen_imgs[gen_label == c]
    # shuffle(gen_imgs)
    # gen_imgs = gen_imgs[:1000]

    real_imgs = np.load('x_val.npy')
    real_label = np.load('y_val.npy')
    real_imgs = real_imgs[real_label == c]
    # shuffle(real_imgs)
    real_imgs = real_imgs[:1000]

    # real_imgs = real_imgs.astype('float32') / 255.

    # resize images
    gen_imgs = scale_images(gen_imgs, (299,299,3))
    real_imgs = scale_images(real_imgs, (299,299,3))
    print('Scaled', gen_imgs.shape, real_imgs.shape)

    # preprocess images
    gen_imgs = preprocess_input(gen_imgs)
    real_imgs = preprocess_input(real_imgs)

    # calculate fid
    fid = calculate_fid(model, gen_imgs, real_imgs)
    print('FID(%d): %.3f' % (c, fid))
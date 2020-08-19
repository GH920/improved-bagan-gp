# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50, imagenet_utils

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x, y = np.load('x_train.npy')[:1000], np.load('y_train.npy')[:1000]

n_classes = len(np.unique(y))
inputShape = x[0].shape

# GAN based augmentation
gen_path = 'bagan_gp_cells_v3_2_epoch100.h5'
generator = load_model(gen_path)
# augmented sample size
aug_size = 100

for c in range(n_classes):
    sample_size = aug_size
    label = np.ones(sample_size) * c
    noise = np.random.normal(0, 1, (sample_size, generator.input_shape[0][1]))
    print('Latent dimension:', generator.input_shape[0][1])
    gen_sample = generator.predict([noise, label])
    gen_imgs = (gen_sample*0.5 + 0.5)*255
    x = np.append(x, gen_imgs, axis=0)
    y = np.append(y, label)
    print('Augmented dataset size:', sample_size, 'Total dataset size:', len(y))

x_train, y_train = x, y

x_test, y_test = np.load('x_val.npy'), np.load('y_val.npy')

preprocess = imagenet_utils.preprocess_input
x_train = preprocess(x_train)
x_test = preprocess(x_test)

# %% -------------------------------------- Model Setup ----------------------------------------------------------------

# Get the feature output from the pre-trained model ResNet50
pretrained_model = ResNet50(include_top=False, input_shape=inputShape, weights="imagenet")
for layer in pretrained_model.layers:
    layer.trainable = False
x = pretrained_model.layers[-1].output
x = GlobalAveragePooling2D()(x)
feature_model = Model(pretrained_model.input, x)

# %% -------------------------------------- TSNE Visualization ---------------------------------------------------------
def tsne_plot(encoder):
    "Creates and TSNE model and plots it"
    plt.figure(figsize=(8, 8))
    color = plt.get_cmap('tab10')

    latent = encoder.predict(x_train)

    # latent = embedding.predict([latent, y_train])
    tsne_model = TSNE(n_components=2, init='random', random_state=0)
    new_values = tsne_model.fit_transform(latent)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    x = np.array(x)
    y = np.array(y)
    x_real = x[:-4 * aug_size]
    y_real = y[:-4 * aug_size]
    x_generated = x[-4 * aug_size:]
    y_generated = y[-4 * aug_size:]
    real_label = y_train[:-4 * aug_size]
    generated_label = y_train[-4 * aug_size:]
    loop = 0
    markers = ['o', 'x']
    for x, y, l in [(x_real, y_real, real_label), (x_generated, y_generated, generated_label)]:
        marker = markers[loop]
        loop += 1
        for c in range(n_classes):
            plt.scatter(x[l == c], y[l == c], marker=marker, c=np.array([color(c)]), label='%d' % c)
    plt.legend()
    plt.show()

tsne_plot(feature_model)

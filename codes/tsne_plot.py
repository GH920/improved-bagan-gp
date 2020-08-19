# %% --------------------------------------- Load Packages -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model

# %% --------------------------------------- Load Models ---------------------------------------------------------------
x_train, y_train = np.load('x_train.npy'), np.load('y_train.npy')
n_classes = len(np.unique(y_train))
# encoder = load_model('bagan_encoder.h5')
encoder = load_model('bagan_gp_encoder1.h5')
embedding = load_model('bagan_gp_embedding1.h5')

# %% --------------------------------------- TSNE Visualization --------------------------------------------------------
def tsne_plot(encoder):
    "Creates and TSNE model and plots it"
    plt.figure(figsize=(8, 8))
    color = plt.get_cmap('tab10')

    latent = encoder.predict(x_train)  # with Encoder
    # latent = embedding.predict([latent, y_train])  ## with Embedding model
    tsne_model = TSNE(n_components=2, init='random', random_state=0)
    new_values = tsne_model.fit_transform(latent)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    x = np.array(x)
    y = np.array(y)

    for c in range(n_classes):
        plt.scatter(x[y_train==c], y[y_train==c], c=np.array([color(c)]), label='%d' % c)
    plt.legend()
    plt.show()

tsne_plot(encoder)
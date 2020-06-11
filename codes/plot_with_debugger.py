import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 10, 10)
# y = x**2
#
# plt.plot(x, y)
# plt.show()

imgs = np.load('x_train.npy')
labels = np.load('y_train.npy')
def plot_type(imgs, type=1, r=2, c=4):
    imgs_type = imgs[labels == type]
    fig, axs = plt.subplots(r, c)
    fig.suptitle('Type %d Cells' % type)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs_type[cnt])
            axs[i,j].axis('off')
            cnt += 1
    plt.show()
plot_type(imgs, type=0)
plot_type(imgs, type=1)
plot_type(imgs, type=2)
plot_type(imgs, type=3)
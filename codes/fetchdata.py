# %% --------------------------------------- Load Packages -------------------------------------------------------------
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %% --------------------------------------- Data Prep -----------------------------------------------------------------
# Download data
if "train" not in os.listdir():
    os.system("cd ~/Capstone")
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
    os.system("unzip train.zip")

# Read data
DIR = 'train/'

train = [f for f in os.listdir(DIR)]
train_sorted = sorted(train, key=lambda x: int(x[5:-4]))
imgs = []
texts = []
resize_to = 64
for f in train_sorted:
    if f[-3:] == 'png':
        imgs.append(cv2.resize(cv2.imread(DIR + f), (resize_to, resize_to)))
    else:
        texts.append(open(DIR + f).read())

imgs = np.array(imgs)
texts = np.array(texts)

le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
labels = le.transform(texts)

# Splitting
SEED = 42
x_train, x_val, y_train, y_val = train_test_split(imgs, labels,
                                                  random_state=SEED,
                                                  test_size=0.2,
                                                  stratify=labels)
print(x_train.shape, x_val.shape)

# %% --------------------------------------- Save as .npy --------------------------------------------------------------
# Save
np.save("x_train.npy", x_train); np.save("y_train.npy", y_train)
np.save("x_val.npy", x_val); np.save("y_val.npy", y_val)
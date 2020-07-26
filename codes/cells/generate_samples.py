import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def gen_samples(n=1000, celltype=1):
    gen_path = 'gan_generator_mlp_5000_type%d.h5' % celltype
    gen = load_model(gen_path)

    sample_size = n
    noise = np.random.normal(0, 1, (sample_size, 100))
    gen_sample = gen.predict(noise)

    np.save('gen_samples_mlp_type%d' % celltype, gen_sample)

gen_samples(1000, 1)
gen_samples(1000, 2)
gen_samples(1000, 3)
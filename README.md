### Cite this paper if it is helpful, thanks!
Huang, G., Jafari, A.H. Enhanced balancing GAN: minority-class image generation. Neural Comput & Applic (2021). https://doi.org/10.1007/s00521-021-06163-8

# Improved-Balancing-GAN-Minority-class-Image-Generation

### Abstract
Generative adversarial networks (GANs) are one of the most powerful generative models, but always require a large and balanced dataset to train. Traditional GANs are not applicable to generate minority-class images in a highly imbalanced dataset. Balancing GAN (BAGAN) is proposed to mitigate this problem, but it is unstable when images in different classes look similar, e.g. flowers and cells. In this work, we propose a supervised autoencoder with an intermediate embedding model to disperse the labeled latent vectors. With the improved autoencoder initialization, we also build an architecture of BAGAN with gradient penalty (BAGAN-GP). Our proposed model overcomes the unstable issue in original BAGAN and converges faster to high quality generations. Our model achieves high performance on the imbalanced scale-down version of MNIST Fashion, CIFAR-10, and one small-scale medical image dataset.

### Problem statement
Although BAGAN proposed an autoencoder initialization to stabilize the GAN training, sometimes the performance of BAGAN is still unstable especially on medical image datasets. Medical image datasets are always: 1. highly imbalanced due to the rare pathological cases, 2. hard to distinguish the difference among classes. As shown in <a href="https://arxiv.org/pdf/1803.09655.pdf">BAGAN paper</a>, the imbalanced `Flowers` dateset has many similar classes so that BAGAN performs not well. In our experiments, BAGAN fails to generate good samples on a small-scale medical image dataset. We consider that the encoder fails to separate images by class when translating them into latent vectors. Our objective of this work is to generate minority-class images in high quality even with a small-scale imbalanced dataset. Our contributions are:  
  - We improve the loss function of BAGAN with gradient penalty and build the corresponding architecture of the generator and the discriminator (BAGAN-GP).
  - We propose a novel architecture of autoencoder with an intermediate embedding model, which helps the autoencoder learn the label information directly.
  - We discuss the drawbacks of the original BAGAN and exemplify performance improvements over the original BAGAN and demonstrate the potential reasons.

### Dataset
1) `MNIST Fashion` and `CIFAR-10`. We also create an imbalanced version for these two datasets.  

|`MNIST Fashion`	| T-shirt |	Trouser |	Pullover  |	Dress |	Coat  |	Sandal  |	Shirt |	Sneaker |	Bag |	Boot  |
| --- | --- |--- | --- |--- |--- | ---| ---| ---| ---| ---|
|Balanced|	4231  |	4165  |	4199  |	4211  |	4185  |	4217  |	4189  |	4241  |	4175  |	4187  |
|Imbalanced|	4166  |	73  |	139 |	210 |	287 |	370 |	422 |	387 |	545 |	651 |  

- MNIST-Fashion (imbalanced-version) with BAGAN-GP
<img src='train_results/imbalanced_mnist_bagan_gp.gif' width='600px'>


|`CIFAR-10`	|Airplane|	Automobile	|Bird	|Cat|	Deer|	Dog	|Frog|	Horse|	Ship	|Truck|
| --- | --- |--- | --- |--- |--- | ---| ---| ---| ---| ---|
|Balanced|	3527	|3523|	3500|	3458|	3563|	3455|	3535|	3509	|3453	|3476|
|Imbalanced|3490|	71|	130|	221|	269|	349	|435|	485|	572|	628|

2) Small-scale imbalanced medical image dataset: `Cells`. Download it with the link `wget https://storage.googleapis.com/exam-deep-learning/train.zip`.  

|`Cells`	|red blood cell (normal)|	ring|	schizont|	trophozoite|
| --- | --- |--- | --- |--- |
|Train| 5600|	292|106|	887|
|Test| 1400	|73	|27| 222|

### Networks
Some neural networks we've referred to in the work:  
1) Generative Adversarial Networks (GANs): BAGAN, WGAN-GP, DRAGAN, ACGAN.  
2) Autoencoder.  
3) Pre-trained networks: ResNet50, Inception V3.

### Framework
[![Python 3.6](https://img.shields.io/badge/Python-3.7-blue.svg)](#)  
1) Keras
2) TensorFlow (2.2)

### Metrics
Fréchet Inception Distance  
<a href="https://www.codecogs.com/eqnedit.php?latex=FID=\ensuremath{\Vert}\mu_r-\mu_g\ensuremath{\Vert}^{2}&plus;Tr\left(\Sigma_r&plus;\Sigma_g-2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?FID=\ensuremath{\Vert}\mu_r-\mu_g\ensuremath{\Vert}^{2}&plus;Tr\left(\Sigma_r&plus;\Sigma_g-2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)" title="FID=\ensuremath{\Vert}\mu_r-\mu_g\ensuremath{\Vert}^{2}+Tr\left(\Sigma_r+\Sigma_g-2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)" /></a>  
FID calculates the feature-level distance between the generated sample distribution and the real sample distribution by Inception V3 network. A smaller FID means the generated distribution is closer to the real distribution. 


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

- CIFAR-10 (balanced-version) with BAGAN-GP
<img src='train_results/training_demo_bagan_gp_cifar.gif' width='600px'>

2) Small-scale imbalanced medical image dataset: `Cells`. Download it with the link `wget https://storage.googleapis.com/exam-deep-learning/train.zip`.  

|`Cells`	|red blood cell (normal)|	ring|	schizont|	trophozoite|
| --- | --- |--- | --- |--- |
|Train| 5600|	292|106|	887|
|Test| 1400	|73	|27| 222|

- Cells with BAGAN-GP
<img src='train_results/training_demo_bagan_gp.gif' width='400px'>

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

### Reference materials
[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.

[3] Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C. Courville. "Improved training of wasserstein gans." In Advances in neural information processing systems, pp. 5767-5777. 2017.

[4] Zhu, Jun-Yan, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired image-to-image translation using cycle-consistent adversarial networks." In Proceedings of the IEEE international conference on computer vision, pp. 2223-2232. 2017.

[5] Almahairi, Amjad, Sai Rajeswar, Alessandro Sordoni, Philip Bachman, and Aaron Courville. "Augmented cyclegan: Learning many-to-many mappings from unpaired data." arXiv preprint arXiv:1802.10151 (2018).

[6] Karras, Tero, Timo Aila, Samuli Laine, and Jaakko Lehtinen. "Progressive growing of gans for improved quality, stability, and variation." arXiv preprint arXiv:1710.10196 (2017).

[7] Zhang, Han, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 (2018).

[8] Shaham, Tamar Rott, Tali Dekel, and Tomer Michaeli. "Singan: Learning a generative model from a single natural image." In Proceedings of the IEEE International Conference on Computer Vision, pp. 4570-4580. 2019.

[9] Gui, Jie, Zhenan Sun, Yonggang Wen, Dacheng Tao, and Jieping Ye. "A review on generative adversarial networks: Algorithms, theory, and applications." arXiv preprint arXiv:2001.06937 (2020).

[10] Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." In International conference on machine learning, pp. 2642-2651. 2017.

[11] Huang, Sheng-Wei, Che-Tsung Lin, Shu-Ping Chen, Yen-Yi Wu, Po-Hao Hsu, and Shang-Hong Lai. "Auggan: Cross domain adaptation with gan-based data augmentation." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 718-731. 2018.

[12] Mariani, Giovanni, Florian Scheidegger, Roxana Istrate, Costas Bekas, and Cristiano Malossi. "Bagan: Data augmentation with balancing gan." arXiv preprint arXiv:1803.09655 (2018).

[13] Shin, Hoo-Chang, Neil A. Tenenholtz, Jameson K. Rogers, Christopher G. Schwarz, Matthew L. Senjem, Jeffrey L. Gunter, Katherine P. Andriole, and Mark Michalski. "Medical image synthesis for data augmentation and anonymization using generative adversarial networks." In International workshop on simulation and synthesis in medical imaging, pp. 1-11. Springer, Cham, 2018.

[14] Kazeminia, Salome, Christoph Baur, Arjan Kuijper, Bram van Ginneken, Nassir Navab, Shadi Albarqouni, and Anirban Mukhopadhyay. "GANs for medical image analysis." arXiv preprint arXiv:1809.06222 (2018).

[15] Srivastava, Akash, Lazar Valkov, Chris Russell, Michael U. Gutmann, and Charles Sutton. "Veegan: Reducing mode collapse in gans using implicit variational learning." In Advances in Neural Information Processing Systems, pp. 3308-3318. 2017.

[16] Luo, Yun, and Bao-Liang Lu. "EEG data augmentation for emotion recognition using a conditional wasserstein GAN." In 2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pp. 2535-2538. IEEE, 2018.

[17] Shorten, Connor, and Taghi M. Khoshgoftaar. "A survey on image data augmentation for deep learning." Journal of Big Data 6, no. 1 (2019): 60.

[18] Dowson, D. C., and B. V. Landau. "The Fréchet distance between multivariate normal distributions." Journal of multivariate analysis 12, no. 3 (1982): 450-455.

[19] Barratt, Shane, and Rishi Sharma. "A note on the inception score." arXiv preprint arXiv:1801.01973 (2018).

[20] Naveen Kodali, Jacob Abernethy, James Hays, and Zsolt Kira. How to train your DRAGAN. arXiv preprint arXiv:1705.07215, 2017.

[21] Qi, Guo-Jun. "Loss-sensitive generative adversarial networks on lipschitz densities." International Journal of Computer Vision 128, no. 5 (2020): 1118-1140.

[22] Zhao, Junbo, Michael Mathieu, and Yann LeCun. "Energy-based generative adversarial network." arXiv preprint arXiv:1609.03126 (2016).

[23] Berthelot, David, Thomas Schumm, and Luke Metz. "Began: Boundary equilibrium generative adversarial networks." arXiv preprint arXiv:1703.10717 (2017).

[24] Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein gan." arXiv preprint arXiv:1701.07875 (2017).

[25] Fedus, William, Mihaela Rosca, Balaji Lakshminarayanan, Andrew M. Dai, Shakir Mohamed, and Ian Goodfellow. "Many paths to equilibrium: GANs do not need to decrease a divergence at every step." arXiv preprint arXiv:1710.08446 (2017).

[26] Shin, Hoo-Chang, Neil A. Tenenholtz, Jameson K. Rogers, Christopher G. Schwarz, Matthew L. Senjem, Jeffrey L. Gunter, Katherine P. Andriole, and Mark Michalski. "Medical image synthesis for data augmentation and anonymization using generative adversarial networks." In International workshop on simulation and synthesis in medical imaging, pp. 1-11. Springer, Cham, 2018.


# Improved-Balancing-GAN-Minority-class-Image-Generation

### Problem statement
Although Balancing GAN (BAGAN) proposed an autoencoder initialization to stabilize the GAN training, sometimes the performance of BAGAN is still unstable especially on medical image datasets. Medical image datasets are always: 1. highly imbalanced due to the rare pathological cases, 2. hard to distinguish the difference among classes. As shown in <a href="https://arxiv.org/pdf/1803.09655.pdf">BAGAN paper</a>, the imbalanced `Flowers` dateset has many similar classes so that BAGAN performs not well. In our experiments, BAGAN fails to generate good samples on a small-scale medical image dataset. We consider that the encoder fails to separate images by class when translating them into latent vectors. Our objective of this work is to generate minority-class images in high quality even with a small-scale imbalanced dataset. Our contributions are:  
  - We improve the loss function of BAGAN with gradient penalty and build the corresponding architecture of the generator and the discriminator (BAGAN-GP).
  - We propose a novel architecture of autoencoder with an intermediate embedding model, which helps the autoencoder learn the label information directly.
  - We discuss the drawbacks of the original BAGAN and exemplify performance improvements over the original BAGAN and demonstrate the potential reasons.

### Dataset
1) `MNIST Fashion` and `CIFAR-10`. We also create an imbalanced version for these two datasets.  

|`MNIST Fashion`	| T-shirt |	Trouser |	Pullover  |	Dress |	Coat  |	Sandal  |	Shirt |	Sneaker |	Bag |	Boot  |
| --- | --- |--- | --- |--- |--- | ---| ---| ---| ---| ---|
|Balanced|	4231  |	4165  |	4199  |	4211  |	4185  |	4217  |	4189  |	4241  |	4175  |	4187  |
|Imbalanced|	4166  |	73  |	139 |	210 |	287 |	370 |	422 |	387 |	545 |	651 |  

|`CIFAR-10`	|Airplane|	Automobile	|Bird	|Cat|	Deer|	Dog	|Frog|	Horse|	Ship	|Truck|
| --- | --- |--- | --- |--- |--- | ---| ---| ---| ---| ---|
|Balanced|	3527	|3523|	3500|	3458|	3563|	3455|	3535|	3509	|3453	|3476|
|Imbalanced|3490|	71|	130|	221|	269|	349	|435|	485|	572|	628|

2) Small-scale imbalanced medical image dataset: Red blood cells `wget https://storage.googleapis.com/exam-deep-learning/train.zip`.  
|`Cells`	|“red blood cell”|	“ring”|	“schizont”|	“trophozoite”|
| --- | --- |--- | --- |--- |
|Train|5600|	292	|106|	887|
|Test|1400	|73	|27|	|222|

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
<img src="https://render.githubusercontent.com/render/math?math=FID=\ensuremath{\Vert}\mu_r-\mu_g\ensuremath{\Vert}^{2}%BTr\left(\Sigma_r%B\Sigma_g-2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)">
```math
FID=\ensuremath{\Vert}\mu_r-\mu_g\ensuremath{\Vert}^{2}+Tr\left(\Sigma_r+\Sigma_g-2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)
```
#### Evaluation of GAN model: 
Inception Score: measure the image quality.  
Multi-scale Structural Similarity (MS-SSIM): measure the diversity and avoid mode collapse.  
#### Evaluation of classifiers: 
Area under the ROC curve (AUC). In medical diagnosis, we care more about finding out all the positive cases even if some images are misclassified into positive labels. (i.e. sensitivity or recall rate).  
### Reference materials
#### GAN-based Augmentation
Huang, Sheng-Wei, Che-Tsung Lin, Shu-Ping Chen, Yen-Yi Wu, Po-Hao Hsu, and Shang-Hong Lai. "Auggan: Cross domain adaptation with gan-based data augmentation." In Proceedings of the European Conference on Computer Vision (ECCV), pp. 718-731. 2018.  
Shin, Hoo-Chang, Neil A. Tenenholtz, Jameson K. Rogers, Christopher G. Schwarz, Matthew L. Senjem, Jeffrey L. Gunter, Katherine P. Andriole, and Mark Michalski. "Medical image synthesis for data augmentation and anonymization using generative adversarial networks." In International workshop on simulation and synthesis in medical imaging, pp. 1-11. Springer, Cham, 2018.  
Kazeminia, Salome, Christoph Baur, Arjan Kuijper, Bram van Ginneken, Nassir Navab, Shadi Albarqouni, and Anirban Mukhopadhyay. "GANs for medical image analysis." arXiv preprint arXiv:1809.06222 (2018).  
Han, Changhee, Kohei Murao, Tomoyuki Noguchi, Yusuke Kawata, Fumiya Uchiyama, Leonardo Rundo, Hideki Nakayama, and Shin'ichi Satoh. "Learning more with less: Conditional PGGAN-based data augmentation for brain metastases detection using highly-rough annotation on MR images." In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 119-127. 2019.  
#### Other Augmentation (Semi-Supervised Learning)
He, Tong, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. "Bag of tricks for image classification with convolutional neural networks." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 558-567. 2019.  
Berthelot, David, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A. Raffel. "Mixmatch: A holistic approach to semi-supervised learning." In Advances in Neural Information Processing Systems, pp. 5050-5060. 2019.  
#### GANs and Literature Review
Gui, Jie, Zhenan Sun, Yonggang Wen, Dacheng Tao, and Jieping Ye. "A review on generative adversarial networks: Algorithms, theory, and applications." arXiv preprint arXiv:2001.06937 (2020).  
Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. "Generative adversarial nets." In Advances in neural information processing systems, pp. 2672-2680. 2014.  
Zhang, Han, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. "Self-attention generative adversarial networks." arXiv preprint arXiv:1805.08318 (2018).  
Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C. Courville. "Improved training of wasserstein gans." In Advances in neural information processing systems, pp. 5767-5777. 2017.  
Karras, Tero, Timo Aila, Samuli Laine, and Jaakko Lehtinen. "Progressive growing of gans for improved quality, stability, and variation." arXiv preprint arXiv:1710.10196 (2017).  
Shaham, Tamar Rott, Tali Dekel, and Tomer Michaeli. "Singan: Learning a generative model from a single natural image." In Proceedings of the IEEE International Conference on Computer Vision, pp. 4570-4580. 2019.  
### Tentative work
1) Use GANs to generate a group of images. -> Apply a prior classifier to soft label these images. -> send true data and generated data into the model.  
2) Train a GAN with the label-0 data (healthy) -> get the D as a pretrained model -> train a GAN with the label-1 data -> generated data -> send all data to train the pretrained D.  

# Kaggle-TGS-Salt-Identification-Challenge

This Repository is a part of my model in the TGS Salt Identification Challenge. https://www.kaggle.com/c/tgs-salt-identification-challenge



# Description:

This is an image segmentation challenge. The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. In particular, in this competition, we are given seismic images (see definition of seismic image: http://www.cpeo.org/techtree/ttdescript/seisim.htm), and our mission is to find where is salt in the picture (find more information in  https://www.kaggle.com/c/tgs-salt-identification-challenge). 

# Difficulties

The images of this competition are seismic images and the data isn't good (either the quality or the quantity), we have just 4000 images (include more than 500 bad images). Some of them are blurred and are made the brightness. Because of the data shortage, we need to do argumentation to get more images. The seismic images make us hard to do augmentation. Only the left-right flip and shifting augmentation are meaning. 

# Model
To approach this challenge, I used the U-net model which was developed by Olaf Ronneberger et al. for BioMedical Image Segmentation. The architecture contains two paths. The first path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. In the encoder part, I used the Resnet 34 model (with some modifications). I tried several models in the encoder part (se-resnet 50, dense net, ...). Finally, Resnet 34 gave to me the best performance. Observing that some of the images are zoomed in, then I used Hypercolumns architecture, see https://arxiv.org/pdf/1411.5752.pdf. 



# Training params

I use Stochastic Gradient Descent with Warm Restarts, see https://arxiv.org/pdf/1705.08790.pdf 

The Lovasz loss is used in this comps, see https://arxiv.org/pdf/1705.08790.pdf.   

# Augmentation

 The following augumentations work for me: 

 

 Left-right flip 

 Small rotation (-10,10) degrees

 Brightness, Contrast, 

 Shift, Scaling

 # Test time augment

 Using only left-right flip for the test time augmentation

 # Ensemble

 I used the Jacarrad score to make assembling. Maybe this is one of the main difference with other competitors. Instead of taking the average of the models, I calculated the Jaccard score among them, this is one of the technique I have learned. 

 # What does not work

 Dilated convolution ( from the smallness of the size of data (101x101))
 

## References
[1]. [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381v2)  
[2]. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)  
[3]. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v2)  
[4]. [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://arxiv.org/abs/1712.02616v2)  
[5]. [Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767v2)  
[6]. [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579v1)  
[7]. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507v1)  
[8]. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1)  
[9]. [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)  
[10]. [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407v1)  
[11]. [A mixed-scale dense convolutional neural network for image analysis](https://slidecam-camera.lbl.gov/static/asset/PNAS.pdf)  
[12]. [Dual Path Networks](https://arxiv.org/abs/1707.01629v2)  
[13]. [Wide Residual Networks](https://arxiv.org/abs/1605.07146v4)  
[14]. [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)  
[15]. [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/abs/1711.09224v1)  
[16]. [Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes](https://arxiv.org/abs/1611.08323v2)  
[17]. [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585v1)  
[18]. [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983v5)  
[19]. [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186v6)  
[20]. [Group Normalization](https://128.84.21.199/abs/1803.08494v1)  
[21]. [Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904v1)  
[22]. [ExFuse: Enhancing Feature Fusion for Semantic Segmentation](https://arxiv.org/abs/1804.03821v1)  
[23]. [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790v2)  
[24]. [Vortex Pooling: Improving Context Representation in Semantic Segmentation](https://arxiv.org/abs/1804.06242v1)  

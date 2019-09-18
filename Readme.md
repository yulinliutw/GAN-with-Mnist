# Basic GAN with Mnist
---
The **Generative Adversarial Networks(GAN)** is a interesting and useful technology deep learning, it can use to synthetic the data and learn the data characteristic from the target domain. The model just take the random noise as the input, and it can output some synthetic image which the content is look like the real image. (we can assign which content we prefer the model to learn during training)
In this implement, I use it to synthetic the Mnist, the result shows how powerful of this technology.
It is authored by **YU LIN LIU**.

### Table of Contents
- <a href='#model-architecture'>Model Architecture</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#evaluation'>Evaluation</a>
- <a href='#performance'>Performance</a>
- <a href='#reference'>Reference</a>

## Model Architecture
---
The Network Architecture is exactly same as in **infoGAN**, I jsut implement it in the pytorch code.
For the loss function and training strategy, it follow the original **GAN** paper and do some simple modify. 

## Installation
---
- Install the [PyTorch](http://pytorch.org/) by selecting your environment on the website.
- Clone this repository.

## Datasets
---
#### MNIST 
This dataset is a large database of handwritten digits that is commonly used for training various image processing systems. 
Current time, the pytorch library can directly provide this dataset, in my setting, I write the *load_data.py* it can load the data for training and evaluation, just check the *load_data.py*, *train.py* and *eval.py*, it can help you to know how to use it.  
**For the detail of this dataset in pytorch, you can check this** [link](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist).

## Training
---
- Open the *train.py* and check the *argparse* setting to understand the training parameters.
- Using the argparse for training parameter setting.
	* Note: we provide the pretrain weight in *./better_weight*, you can load it by setting the *load_weight_dir* parameter.
- Start the training.
```Shell
python train.py
```

## Evaluation
---
This action will show some result from model and the training history of the model. 

- Open the *eval.py* and check the argparse setting to understand the evaluation parameters.
- Using the argparse for evaluation parameter setting.
- Start the evualation.
```Shell
python eval.py
```

## Performance
---
- I train this model about 40 epoch.
- The training history -- Real/fake predict loss :   
This figure shows the training history of the discriminator, the **blue curve** shows how many % prediction is fail(predict it as fake) when input the real image, the **orange curve** shows how many % prediction is fail(predict it as real) when input the fake image. We can see that the discriminator have better performance in the early epoch, but after 10 epoch, the loss become higher, that's because the generator is begin to know how to synthetic the meaningful image, so the discriminator is confuse with some input, it doesn't really know it come from real.  

<p align="center"><img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" " height='230px' width='230px'></p>

- The training history -- GAN/Dis loss :	
  This figure shows the training history of the generator and discriminator, the loss curve is caculate by the loss function which using in training. We can see the converge tendency the GAN, two model will find the balance finally.
  
<p align="center"><img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" "  height='230px' width='230px'></p>

- The training history -- synthetic result durning each epoch(5,10,20,40) : 

|         | Result          | 
| ------------- |:-------------:| 
| **Epoch5**     |  <img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" "  height='230px' width='230px'> | 
| **Epoch10**   |  <img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" "  height='230px' width='230px'> |  
| **Epoch20**  |  <img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" "  height='230px' width='230px'>| 
| **Epoch40**  |  <img src="https://github.com/yulinliutw/Basic-GAN-with-Mnist/blob/master/doc/result.png" alt=" "  height='230px' width='230px'>| 

## Reference
---
####paper
- [GAN](https://arxiv.org/abs/1406.2661)
- [infoGAN](https://arxiv.org/abs/1606.03657)

####code
- [G and D in tensorflow-generative-model-collection](https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/GAN.py)
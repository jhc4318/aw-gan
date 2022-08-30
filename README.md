# AW-GAN

This is the implementation code for AW-GAN which has been built on top of DAGAN (Deep De-Aliasing Generative Adversarial Network), referenced below. Compared to DAGAN:

- an additional policy gradient network (pg_net) has been added to models.py
- a modified training routine (train_mod.py) has been created to allow for training with the pg_net
- helper functions appended to utils.py
- parameters for pg_net added to config.py
- loss_parser.py and loss_visualiser.py created to aid with visualisation of data

This was written with Python 3.5 and a requirements.txt has been provided for ease of use. CUDA CuDNN is required, with compatibility with Tensorflow v1.1.0.

# Instructions for usage

Shell scripts have been provided for use with Imperial's GPU cluster on Slurm. This assumes existence of a conda virtual environment "dagan". Instructions below are a modified version of DAGAN's.

1. Data preparation

	1) Data was used from the 2013 MICCAI Grand Challenge dataset ([link](https://my.vanderbilt.edu/masi/workshops/)). Registration is required.
	2) Download data/MICCAI13_SegChallenge/Training_100 and data/MICCAI13_SegChallenge/Testing_100 for training and testing data. Modify training paths in config.py and data_loader.py as necessary. The randomly selected sample used for training is given in dataset_name_list.txt.
	3) Run "python data_loader.py".
	
2. Download pretrained VGG16 model

    1) Download 'vgg16_weights.npz' from [this link](http://www.cs.toronto.edu/~frossard/post/vgg16/)
    2) Save 'vgg16_weights.npz' into 'trained_model/VGG16'
    
3. Train model
    1) run 'CUDA_VISIBLE_DEVICES=0 python train_mod.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively:
    - MODEL: choose from 'unet' or 'unet_refine'
    - MASK: choose from 'gaussian1d', 'gaussian2d', 'poisson2d'
    - MASKPERC: choose from '10', '20', '30', '40', '50' (percentage of mask)
    
4. Test trained model
    1) run 'CUDA_VISIBLE_DEVICES=0 python test.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively (as above).

# Reference to DAGAN

```
@article{yang2018_dagan,
	author = {Yang, Guang and Yu, Simiao and Dong, Hao and Slabaugh, Gregory G. and Dragotti, Pier Luigi and Ye, Xujiong and Liu, Fangde and Arridge, Simon R. and Keegan, Jennifer and Guo, Yike and Firmin, David N.},
	journal = {IEEE Trans. Med. Imaging},
	number = 6,
	pages = {1310--1321},
	title = {{DAGAN: deep de-aliasing generative adversarial networks for fast compressed sensing MRI reconstruction}},
	volume = 37,
	year = 2018
}
```

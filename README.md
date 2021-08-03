# whole_slide_images

- big_nephro_dataset.py contains the dataset, to be used with pytorch dataloaders
- 10Y_main.py is to be called in order to train neural networks. In particular, the MyResNet class is a neural network to evaluate multiple images together, for example several patches from a whole-slide-image.
- 10Y_inference.py is to be called to perform inference with ensemble on an already trained network.
- bignephroqpdata_regions.py contains several processing function used to polish the dataset and extrapolate patches from whole-slide-images. In particular, the function processing_pipeline might be useful.


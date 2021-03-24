# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Attributions
- https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/- https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
- https://seaborn.pydata.org/introduction.html- https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
- https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
- https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
- https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
- https://discuss.pytorch.org/t/runtimeerror-expected-object-of-type-torch-floattensor-but-found-type-torch-cuda-floattensor-for-argument-2-weight/27483
- https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/21
- https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu

## Commands
### train.py
```
python train.py -h
usage: train.py [-h] [-d DATA_DIR] [-s SAVE_DIR] [-a ARCH] [-lr LEARNING_RATE] [-hu HIDDEN_UNITS] [-e EPOCHS] [-g]

This is a script to train vgg16.

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_DIR, --data-dir DATA_DIR
                        This is the folder containing the flower images. If not set, flowers is used.
  -s SAVE_DIR, --save-dir SAVE_DIR
                        The destination folder name to save the model (default is 'output').
  -a ARCH, --arch ARCH  An architecture type from torchvision models. Valid options are: vgg16, vgg19. (default is vgg16)
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Gradient descent learning rate. Default is 0.001.
  -hu HIDDEN_UNITS, --hidden-units HIDDEN_UNITS
                        Number of hidden units of the input classifier layer. Default is 4096.
  -e EPOCHS, --epochs EPOCHS
                        Epoch (repetitions) for training. Default is 3.
  -g, --gpu             Enable GPU mode (default is CPU).

```

### predict.py
```
python predict.py -h
usage: predict.py [-h] -i INPUT -c CHECKPOINT [-tk TOP_K] -cn CATEGORY_NAMES [-g]

This tool predicts the class of an Image.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        This is the path to an image
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        The path to a stored checkpoint.
  -tk TOP_K, --top-k TOP_K
                        The number of most likely classes to display (default is 5).
  -cn CATEGORY_NAMES, --category-names CATEGORY_NAMES
                        Path to json file that maps classes to human-friendly names.
  -g, --gpu             Enable GPU mode (default is CPU).
```
python predict.py -i flowers/valid/77/image_00245.jpg -c output/model_20210322_143902.pth -tk 3 -cn cat_to_name.json  -g
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

## Command
python predict.py -i flowers/valid/77/image_00245.jpg -c output/model_20210322_143902.pth -tk 3 -cn cat_to_name.json  -g
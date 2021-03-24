# -*- coding: utf-8 -*-

import numpy as np
import json
from PIL import Image
from math import ceil
import torch
from math import ceil
from train import Classifier
from torchvision import models

import argparse

DEFAULT_TOPK=5


class PredictFlowerName:

    def __init__(self, args):

        # Load category names
        print('Loading category names...')
        cat_to_name_file = 'cat_to_name.json'
        # if args.category_names is not None:
        cat_to_name_file = args.category_names
        with open(cat_to_name_file, 'r') as f:
            cat_to_name = json.load(f)

        # Load model checkpoint trained with OLDtrain.py
        print('Loading model definition from file ' + cat_to_name_file)
        model_from_file = self.recreateModel(args.checkpoint)

        # Use GPU if requested
        current_device = self.use_gpu(model_from_file, args.gpu)

        # Predict image class
        print('Predicting image class...\n')

        image_path = args.input
        probs, classes = self.predict(image_path, model_from_file, topk=args.top_k)
        names = self.categoryToName(classes)

        # Read the flower name based on the folder
        # print('Image path: ' + image_path)
        folder_number = image_path.split('/')[2]
        # print('Folder number: ' + folder_number)
        label = cat_to_name[folder_number]
        # print('Label: ' + label)

        print('Unknown image location: {}'.format(image_path))
        print('Unknown image category #: {}'.format(folder_number))
        print('Unknown image name: {}'.format(label))
        print('\nShowing {} most likely classes. The correct class is {}'.format(args.top_k, folder_number))

        index = 0
        for i, j, k in zip(probs, classes, names):
            print("{}. flower name: {} | class: {} | likelihood: {}%".format(index + 1, k, j, ceil(i * 100)))
            index = index + 1

    def recreateModel(self, filepath):
        # Load model metadata
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        # Recreate the pretrained base model
        # model = models.vgg16(pretrained=True)
        model = getattr(models, checkpoint['name'])(pretrained=True)

        # Replace the classifier part of the model
        model.classifier = checkpoint['classifier']

        # Rebuild saved state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load class_to_idx
        model.class_to_idx = checkpoint['class_to_idx']

        return model

    def process_image(self, image):
        """
        :param image: string path to an image
        :return: Numpy array
        """

        # Find the shorter side and resize it to 256 keeping aspect ration
        # if the width > height
        if image.width > image.height:
            # Constrain the height to be 256
            image.thumbnail((10000000, 256))
        else:
            # Constrain the width to be 256
            image.thumbnail((256, 10000000))

        # Center crop the image
        crop_size = 224
        left_margin = (image.width - crop_size) / 2
        bottom_margin = (image.height - crop_size) / 2
        right_margin = left_margin + crop_size
        top_margin = bottom_margin + crop_size
        image = image.crop((left_margin, bottom_margin, right_margin, top_margin))

        # Convert values to range of 0 to 1 instead of 0-255
        image = np.array(image)
        image = image / 255

        # Standardize values
        means = np.array([0.485, 0.456, 0.406])
        stds = np.array([0.229, 0.224, 0.225])
        image = (image - means) / stds

        # Move color channels to first dimension as expected by PyTorch
        image = image.transpose(2, 0, 1)

        return image

    def predict(self, image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
           '''

        # TODO: Implement the code to predict the class from an image file

        # Move model into evaluation mode and to CPU
        model.eval()
        model.cpu()

        # Open image
        image = Image.open(image_path)

        # Process image
        image = self.process_image(image)

        # Change numpy array type to a PyTorch tensor
        image = torch.from_numpy(image).type(torch.FloatTensor)

        # Format tensor for input into model
        # (add batch of size 1 to image)

        image = image.unsqueeze(0)

        # Predict top K probabilities
        # Reverse the log conversion
        probs = torch.exp(model.forward(image))
        top_probs, top_labs = probs.topk(topk)
        # print(top_probs)
        # print(top_labs)

        # Convert from Tesors to Numpy arrays
        top_probs = top_probs.detach().numpy().tolist()[0]
        idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}

        # Map tensor indexes to classes
        labs = []
        for label in top_labs.numpy()[0]:
            labs.append(idx_to_class[label])

        return top_probs, labs

    def categoryToName(self, categories, mapper='cat_to_name.json'):
        # Load json file
        with open(mapper, 'r') as f:
            cat_to_name = json.load(f)

            names = []

            # Find flower names corresponding to predicted categories
            for category in categories:
                names.append(cat_to_name[str(category)])

        return names

    def use_gpu(self, model, gpu):

        current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            if gpu:
                print('GPU processing requested and available')
            else:
                print("Use '--gpu' option to enable GPU/CUDA processing.")

        model.to(current_device)
        return current_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This tool predicts the class of an Image.')
    # parser.add_argument('--foo', action='store_true', help='foo help')
    # subparsers = parser.add_subparsers(help='sub-command help')

    parser.add_argument('-i', '--input',
                        help='This is the path to an image',
                        type=str,
                        required=True)

    parser.add_argument('-c', '--checkpoint',
                        help='The path to a stored checkpoint.',
                        type=str,
                        required=True)

    parser.add_argument('-tk', '--top-k', default=DEFAULT_TOPK,
                        help='The number of most likely classes to display (default is ' + str(DEFAULT_TOPK) + ').',
                        type=int,
                        required=False)

    parser.add_argument('-cn', '--category-names',
                        help='Path to json file that maps classes to human-friendly names.',
                        type=str,
                        required=True)

    parser.add_argument('-g', '--gpu',
                        action="store_true",
                        required=False,
                        help='Enable GPU mode (default is CPU).')

    args = parser.parse_args()

    try:
        obj = PredictFlowerName(args)

    except AttributeError:
        print("\nUse -h for instructions.\n")


# python OLDpredict.py -i flowers/valid/77/image_00245.jpg -c output/model_20210322_143902.pth -tk 3 -cn cat_to_name.json  -g
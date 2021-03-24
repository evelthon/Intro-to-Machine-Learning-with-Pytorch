# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
# from torch.optim.lr_scheduler import StepLR

# from workspace_utils import active_session

import torch.nn.functional as F
import time
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import random
import os
import argparse

# import seaborn as sns

# Define variables
DATA_DIR = 'flowers'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/valid'
TEST_DIR = DATA_DIR + '/test'

DEFAULT_DATADIR = DATA_DIR
DEFAULT_SAVEDIR = 'output'
DEFAULT_ARCH='vgg16'
DEFAULT_LEARNINGRATE=0.001
DEFAULT_HIDDENUNITS=4096
DEFAULT_EPOCHS=3

# Dataset controls
IMAGE_SIZE = 224  # Image size in pixels
REDUCTION = 255  # Image reduction to smaller edge
NORM_MEANS = [0.485, 0.456, 0.406]  # Normalized means of the images
NORM_STD = [0.229, 0.224, 0.225]  # Normalized standard deviations of the images
ROTATION = 45  # Degrees for rotation
BATCH_SIZE = 64  # Number of images used in a single pass
SHUFFLE = True  # Randomize image selection

# Argparse configuration
SUPPORTED_ARCHITECTURES = ['vgg13', 'vgg16', 'vgg19']




class ImageClassifier:
    def __init__(self, args):
        # data_dir = args.data_dir
        # save_dir = args.save_dir
        # arch = args.arch
        # learning_rate = args.learning_rate
        # hidden_units = args.hidden_units
        # epochs = args.epochs
        training_loader, validation_loader, testing_loader, training_data = self.get_data(args.data_dir)

        model = self.initialize_model(args.arch)
        print (model)
        print(self.build_classifier(args.hidden_units))
        if model:
            model.classifier = self.build_classifier(args.hidden_units)
        print('Model architecture: \n{}'.format(model))

        current_device = self.use_gpu(model, args.gpu)

        # Define the loss function
        criterion = nn.NLLLoss()

        # Hyperparameters
        drop_out = 0.2

        learning_rate = args.learning_rate

        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        num_of_epochs = args.epochs

        print('Training started...')
        if not args.gpu:
            print('Go brew some coffee! Training on CPU.')

        # Train and validate the neural network classifier
        train_loss, valid_loss, valid_accuracy = self.trainClassifier(
            model, 
            num_of_epochs,
            criterion,
            optimizer,
            training_loader,
            validation_loader,
            current_device
        )

        # Display final summary
        print("Final result \n",
              f"Train loss: {train_loss:.3f} \n",
              f"Test loss: {valid_loss:.3f} \n",
              f"Test accuracy: {valid_accuracy:.3f}")

        print('Training completed.')

        # Save the generated model
        filename = self.saveCheckpoint(model, training_data, args.save_dir)
        print('Model saved in {}.'.format(filename))

    def get_data(self, datadir):

        # Create transforms to apply on image data
        # Next convert image data to sensors and normalize it to make backpropagation more stable
        training_transforms = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomRotation(ROTATION),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEANS, NORM_STD)
        ])

        validation_transforms = transforms.Compose([
            transforms.Resize(REDUCTION),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEANS, NORM_STD)
        ])

        testing_transforms = transforms.Compose([
            transforms.Resize(REDUCTION),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEANS, NORM_STD)
        ])

        # Load and transform image data
        training_data = datasets.ImageFolder(TRAIN_DIR, transform=training_transforms)
        validation_data = datasets.ImageFolder(VALID_DIR, transform=validation_transforms)
        testing_data = datasets.ImageFolder(TEST_DIR, transform= testing_transforms)

        # Use image datasets and the transforms to define the data loaders
        training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=BATCH_SIZE)
        testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=BATCH_SIZE)

        return training_loader, validation_loader, testing_loader, training_data

    def initialize_model(self, arch):

        torch_model = arch if arch in SUPPORTED_ARCHITECTURES else DEFAULT_ARCH
        print(torch_model)
        # Load a pre-trained network
        # https://pytorch.org/docs/stable/torchvision/models.html
        model = getattr(models, torch_model)(pretrained=True)
        model.name = torch_model

        # Freeze model parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        print(model)

        return model

    def build_classifier(self, hidden_units):
        input_size = 25088
        output_size = 102
        # if hidden_units is None:
        #     hidden_units = 4096
        hidden_layers = [hidden_units, 1024]
        # hidden_layers.append(int(hidden_units))
        # hidden_layers.append(1024)
        drop_out = 0.2

        return Classifier(input_size, output_size, hidden_layers, drop_out)

    # Function to validate and test
    def testClassifier(self, model, criterion, testing_loader, current_device):
        # Move the network and data to current hardware config (GPU or CPU)
        model.to(current_device)

        test_loss = 0
        accuracy = 0

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in testing_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(current_device), labels.to(current_device)

            # Forward pass, then backward pass, then update weights
            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
            test_loss += batch_loss.item()

            # Convert to softmax distribution
            ps = torch.exp(log_ps)

            # Compare highest prob predicted class with labels
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            # Calculate accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return test_loss, accuracy

    # A function used for training (and tests with different model hyperparameters)
    def trainClassifier(self, model, num_of_epochs, criterion, optimizer, training_loader, validation_loader, current_device):
        # Move the network and data to current hardware config (GPU or CPU)
        model.to(current_device)

        epochs = num_of_epochs
        steps = 0
        print_every = 1
        running_loss = 0

        # Looping through epochs, each epoch is a full pass through the network
        for epoch in range(epochs):

            # Switch to the train mode
            model.train()

            # Looping through images, get a batch size of images on each loop
            for inputs, labels in training_loader:
                steps += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(current_device), labels.to(current_device)

                # Clear the gradients, so they do not accumulate
                optimizer.zero_grad()

                # Forward pass, then backward pass, then update weights
                log_ps = model(inputs)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Track the loss and accuracy on the validation set to determine the best hyperparameters
            # if steps % print_every == 0:
            # Put in evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():
                # Validate model
                test_loss, accuracy = self.testClassifier(model, criterion, validation_loader, current_device)

            train_loss = running_loss / print_every
            valid_loss = test_loss / len(validation_loader)
            valid_accuracy = accuracy / len(validation_loader)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train loss: {train_loss:.3f} | "
                  f"Test loss: {valid_loss:.3f} | "
                  f"Test accuracy: {valid_accuracy:.3f}")

            running_loss = 0

            # Switch back to the train mode
            model.train()

        # Return last metrics
        return train_loss, valid_loss, valid_accuracy

    def saveCheckpoint(self, model, training_data, savedir=''):
        # Mapping of classes to indices
        model.class_to_idx = training_data.class_to_idx

        # Create model metadata dictionary
        checkpoint = {
            'name': model.name,
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'model_state_dict': model.state_dict()
        }

        # Save to a file
        timestr = time.strftime("%Y%m%d_%H%M%S")
        file_name = 'model_' + timestr + '.pth'

        if not savedir is None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            file_name = os.path.join(savedir, file_name)

        torch.save(checkpoint, file_name)

        return file_name

    def use_gpu(self, model, gpu):

        print(gpu)
        current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            if gpu:
                print('GPU processing requested and available')
            else:
                print("Use '--gpu' option to enable GPU/CUDA processing.")

        model.to(current_device)
        return current_device

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()

        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])

        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a script to train vgg16.')
    # parser.add_argument('--foo', action='store_true', help='foo help')
    # subparsers = parser.add_subparsers(help='sub-command help')
    '''
    
    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
        Choose architecture: python train.py data_dir --arch "vgg13"
        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
        Use GPU for training: python train.py data_dir --gpu

    '''


    parser.add_argument('-d', '--data-dir', default=DEFAULT_DATADIR,
                        help='This is the folder containing the flower images. If not set, ' + DEFAULT_DATADIR + ' is used.',
                        type=str,
                        required=False)

    parser.add_argument('-s', '--save-dir', default=DEFAULT_SAVEDIR,
                        help='The destination folder name to save the model (default is \'' + DEFAULT_SAVEDIR + '\').',
                        type=str,
                        required=False)

    parser.add_argument('-a', '--arch', default=DEFAULT_ARCH,
                        help='An architecture type from torchvision models. Valid options are: vgg16, vgg19. (default is ' + DEFAULT_ARCH + ')',
                        type=str,
                        required=False)

    #arguments for hyperparameters
    # https://stackoverflow.com/questions/42966393/is-it-good-learning-rate-for-adam-method
    parser.add_argument('-lr', '--learning-rate', default=DEFAULT_LEARNINGRATE,
                        help='Gradient descent learning rate. Default is ' + str(DEFAULT_LEARNINGRATE) + '.',
                        type=float,
                        required=False)
    parser.add_argument('-hu', '--hidden-units', default=DEFAULT_HIDDENUNITS,
                        help='Number of hidden units of the input classifier layer. Default is ' + str(DEFAULT_HIDDENUNITS) + '.',
                        type=int,
                        required=False)
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS,
                        help='Epoch (repetitions) for training. Default is ' + str(DEFAULT_EPOCHS) + '.',
                        type=int,
                        required=False)

    # Enable GPU training
    parser.add_argument('-g', '--gpu',
                        action="store_true",
                        required=False,
                        help='Enable GPU mode (default is CPU).')


    args = parser.parse_args()

    try:
        obj = ImageClassifier(args)

    except AttributeError:
        print("\nUse -h for instructions.\n")

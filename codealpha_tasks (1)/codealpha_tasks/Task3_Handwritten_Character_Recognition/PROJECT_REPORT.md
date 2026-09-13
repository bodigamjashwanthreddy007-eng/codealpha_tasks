# Project Report: Handwritten Character Recognition

## 1. Introduction
Handwritten character recognition is a computer vision problem in which a machine-learning model identifies handwritten characters from images.

## 2. Objective
The objective of this project is to recognize handwritten digits using the MNIST dataset and a Convolutional Neural Network.

## 3. Dataset
MNIST contains grayscale images of handwritten digits from 0 to 9. Each image has a size of 28 x 28 pixels.

## 4. Preprocessing
- Pixel values are normalized from 0-255 to 0-1.
- A channel dimension is added for CNN input.
- Training data is divided into training and validation sets.
- A subset is used to keep the internship demonstration practical to train.

## 5. CNN Architecture
The model contains:
- Conv2D layer
- MaxPooling2D layer
- Conv2D layer
- MaxPooling2D layer
- Flatten layer
- Dense layer
- Dropout layer
- Softmax output layer

## 6. Training
The model is trained using the Adam optimizer and sparse categorical cross-entropy loss for 5 epochs.

## 7. Evaluation
The model is evaluated using:
- Test loss
- Test accuracy
- Classification report
- Confusion matrix

Training and validation accuracy/loss graphs are also generated.

## 8. Results
The exact results produced by the local run are stored in `results/model_metrics.csv`.

## 9. Conclusion
This project demonstrates an end-to-end deep-learning workflow for handwritten digit recognition, including image preprocessing, CNN training, evaluation, and prediction visualization.

## 10. Limitation
The project focuses on the MNIST digit dataset and does not recognize arbitrary handwritten words or sentences.

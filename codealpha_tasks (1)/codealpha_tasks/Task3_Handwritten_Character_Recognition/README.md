# Task 3 - Handwritten Character Recognition

## CodeAlpha Machine Learning Internship

### Objective
Recognize handwritten digits using the MNIST dataset and a Convolutional Neural Network (CNN).

### Dataset
The project uses the MNIST handwritten digit dataset. Each image is a 28 x 28 grayscale image representing a digit from 0 to 9.

### Model
A Convolutional Neural Network is used with:
- Convolution layers
- Max pooling
- Fully connected layer
- Dropout
- Softmax output layer

### Workflow
1. Load the MNIST dataset.
2. Normalize image pixel values.
3. Prepare images for CNN input.
4. Split training data into training and validation sets.
5. Train the CNN.
6. Evaluate the model on test data.
7. Generate a classification report and confusion matrix.
8. Generate accuracy/loss curves.
9. Generate sample predictions.

### How to run

```bash
pip install -r requirements.txt
python handwritten_character_recognition.py
```

The first run may download the MNIST dataset through Keras.

### Project Structure

```text
Task3_Handwritten_Character_Recognition/
├── handwritten_character_recognition.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── LINKEDIN_POST.md
├── VIDEO_SCRIPT.md
└── results/
    ├── classification_report.txt
    ├── model_metrics.csv
    ├── confusion_matrix.png
    ├── accuracy_curve.png
    ├── loss_curve.png
    ├── sample_predictions.png
    └── sample_prediction.txt
```

### Note
This project recognizes handwritten digits from 0 to 9. It is an educational internship project and is not a complete handwritten word/sentence recognition system.

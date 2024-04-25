# American Sign Language Recognition

## Project Overview
This project implements a convolutional neural network (CNN) to classify images of American Sign Language (ASL) alphabets. It utilizes TensorFlow and Keras for building and training the model, demonstrating an end-to-end machine learning pipeline from data preprocessing to model evaluation.

## Features
- **Data Preprocessing**: Utilizes `ImageDataGenerator` for image augmentation to enhance the model's robustness.
- **Model Architecture**: A sequential CNN model with layers optimized for image classification tasks.
- **Training**: Includes custom callbacks for dynamic learning rate adjustments.
- **Evaluation**: Measures accuracy, precision, recall, and F1-score on a separate test dataset.
- **Visualization**: Feature maps from the CNN and predictions visualizations.

## Model Architecture
The model consists of several convolutional layers, max pooling, and dropout layers to prevent overfitting, followed by a dense output layer for classification:
- Conv2D with 32 filters
- MaxPooling2D
- Conv2D with 64 filters
- MaxPooling2D
- Conv2D with 128 filters
- MaxPooling2D
- Flatten
- Dense with 512 units
- Dropout (0.5)
- Dense output layer with softmax activation

## How to Run
1. Clone this repository.
2. Ensure you have Python 3.8+, TensorFlow 2.x, and all required libraries installed.
3. Place your dataset in the `asl_alphabet_train` and `asl_alphabet_test` directories accordingly.
4. Run the Jupyter Notebook or Python scripts provided.

### Install Requirements
```bash
pip install -r requirements.txt

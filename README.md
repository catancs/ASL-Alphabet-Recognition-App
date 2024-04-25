## 👋 Welcome to the Streamlit ASL Alphabet Recognition App!

This project includes a Streamlit web application for real-time American Sign Language (ASL) alphabet recognition. The app allows users to upload an image of an ASL alphabet sign, and it will classify the sign and display the predicted label. It utilizes TensorFlow and Keras for building and training the model, demonstrating an end-to-end machine learning pipeline from data preprocessing to model evaluation.


## Features
- **Real-time Recognition**: Upload an image and get instant predictions.
- **User-friendly Interface**: Simple and intuitive design for easy interaction.
- **Data Preprocessing**: Utilizes `ImageDataGenerator` for image augmentation to enhance the model's robustness.
- **Model Architecture**: A sequential CNN model with layers optimized for image classification tasks.
- **Training**: Includes custom callbacks for dynamic learning rate adjustments.
- **Evaluation**: Measures accuracy, precision, recall, and F1-score on a separate test dataset.
- **Visualization**: Feature maps from the CNN and predictions visualizations.

## Technologies Used
- **Python**: Programming language
- **TensorFlow/Keras**: Deep learning framework for model development
- **Streamlit**: Web application framework for deployment
- **Pillow**: Image processing library for handling image uploads

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

## How to Run the Web App
1. Clone this repository.
2. Ensure you have installed requirements listed below.
3. Place your dataset in the `asl_alphabet_train` and `asl_alphabet_test` directories accordingly.
4. Run the streamlit web app using the following command:
```
streamlit run asl_app.py
```
5. Uploand an image with a ASL symbol and see the results.

### Install Requirements
```bash
pip install -r dependencies.txt
```
## Contributions
Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.


## 🚀 Explore the app here and start recognizing ASL alphabet signs with ease!


---

# Handwritten Digits Recognition using MNIST Database

Utilizing Convolutional Neural Networks for Handwritten Text Recognition with the MNIST Dataset.

## General Overview of the MNIST Database
![MnistExamplesModified](https://github.com/BlizzyBastard/MNIST_Project/assets/122042171/6a0a8312-6099-4531-be0d-297b560d017b)

The MNIST database, a Modified National Institute of Standards and Technology database, stands as a pivotal dataset in handwritten digit recognition. It encompasses 60,000 training images and 10,000 testing images, serving as a fundamental resource for training and testing diverse machine learning models.

## Website Functionality
![Screenshot 2023-11-18 at 12-39-01 Handwritten Digit Recognition](https://github.com/BlizzyBastard/MNIST_Project/assets/122042171/2a69355a-cf29-4ca7-a5a2-db548d5d7243)

### Features:
- **Canvas Drawing:** Allows users to create handwritten digits on a canvas.
- **Model Prediction:** Leverages a Flask backend for real-time HTML updates with model predictions.
- **Upload Option:** Provides an alternative for users to upload images for prediction.

The website amalgamates JavaScript for canvas functionality with fundamental HTML and CSS for user interaction.

## Model Training 
The model is constructed using Convolutional Neural Networks (CNNs) with TensorFlow.

### Model Overview
```plaintext
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
conv2d (Conv2D)             (None, 26, 26, 64)        640       
conv2d_1 (Conv2D)           (None, 24, 24, 128)       73856     
max_pooling2d                (None, 12, 12, 128)       0         
dropout                      (None, 12, 12, 128)       0         
conv2d_2 (Conv2D)           (None, 10, 10, 128)       147584    
max_pooling2d_1              (None, 5, 5, 128)         0         
dropout_1                    (None, 5, 5, 128)         0         
flatten                      (None, 3200)              0         
dense                        (None, 256)               819456    
dropout_2                    (None, 256)               0         
dense_1                      (None, 10)                2570      
=================================================================
Total params: 1,044,106 (3.98 MB)
Trainable params: 1,044,106 (3.98 MB)
Non-trainable params: 0
```

### Performance Metrics
- **Accuracy Plot:**
  
  ![accuracy_plot](https://github.com/BlizzyBastard/MNIST_Project/assets/122042171/fb77c4c4-702d-49ee-8509-4c45b82817ef)
- **Loss Plot:**
 
  ![loss_plot](https://github.com/BlizzyBastard/MNIST_Project/assets/122042171/2eb2d679-68e8-4224-a6a8-eb37b259748f)
- **Confusion Matrix:**
  
  ![Confusion_Matrix](https://github.com/BlizzyBastard/MNIST_Project/assets/122042171/13522b8b-68c4-45b5-b7f0-ee464c3f2358)

### Description:
This CNN architecture incorporates convolutional, max-pooling, dropout, and dense layers, totaling around 1.04 million parameters. It demonstrates exceptional proficiency in recognizing handwritten digits, as illustrated by performance metrics.

### Efficiency
The network showcases high efficiency, achieving a remarkable accuracy rate with minimal loss. The plotted accuracy, loss, and confusion matrix affirm its robustness in deciphering handwritten digits accurately.

### Running the Flask App
To launch the Flask app:
1. Set the App Name:
   ```bash
   SET APP_NAME=app.py  # For Windows
   export APP_NAME=app.py  # For MacOS/Linux
   ```

2. Run the Flask App:
   ```bash
   flask run
   ```

Before executing the Flask app, ensure all dependencies are installed and configured appropriately.

---

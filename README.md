# Image-Classification-using-MLP-on-CIFAR-10-Dataset

This project demonstrates how to build a **Multilayer Feed Forward Neural Network (MLP)** to classify images in the **CIFAR-10 dataset**. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

---

## 🎯 Objective

To build and train a **Multilayer Feed Forward Neural Network (MLP)** to classify images from the **CIFAR-10 dataset** into 10 categories:

* **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, and **truck**.

The goal is to classify these images with high accuracy using a neural network approach and evaluate the model's performance.

---

## 📂 Dataset Overview

The CIFAR-10 dataset is a well-known dataset for image classification tasks. It contains:

* **60,000 images**: 50,000 for training and 10,000 for testing
* **10 classes**: Each image belongs to one of the following categories:

  * Airplane
  * Automobile
  * Bird
  * Cat
  * Deer
  * Dog
  * Frog
  * Horse
  * Ship
  * Truck

You can download the dataset using libraries like **Keras** or **TensorFlow**, which provide built-in access to it.

---

## 🛠️ Tools & Technologies

* **Python**
* **TensorFlow / Keras** for building the neural network
* **NumPy**, **Pandas** for data manipulation
* **Matplotlib** for visualizations
* **Scikit-learn** for evaluation metrics

---

## 📊 Project Workflow

### 1. Data Preprocessing

* Load the CIFAR-10 dataset using Keras or TensorFlow
* Normalize pixel values to the range \[0, 1] by dividing by 255
* Flatten the 32x32 images into 1D vectors (e.g., 3072 features per image)
* Split the dataset into training and testing sets (Keras provides this by default)

### 2. Model Architecture

The **Multilayer Feed Forward Neural Network (MLP)** typically consists of:

* **Input Layer**: Flatten the 32x32 image into a 1D vector of 3072 features
* **Hidden Layers**: Use dense (fully connected) layers with ReLU activation
* **Output Layer**: 10 units with softmax activation for classification

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
    Flatten(input_shape=(32, 32, 3)),  # Flatten 32x32x3 images
    Dense(512, activation='relu'),     # Hidden layer with 512 units
    Dense(256, activation='relu'),     # Hidden layer with 256 units
    Dense(10, activation='softmax')    # Output layer with 10 units (classes)
])
```

### 3. Model Compilation & Training

* **Loss function**: `categorical_crossentropy` for multi-class classification
* **Optimizer**: `adam` for efficient training
* **Metrics**: `accuracy`

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### 4. Evaluation & Results

* Evaluate the model on the test set
* Display accuracy and loss graphs
* Confusion matrix and classification report

---

## 📈 Results

| Metric         | Value  |
| -------------- | ------ |
| Accuracy       | \~75%+ |
| Loss           | \~0.85 |
| Inference Time | Fast   |


## 🔧 Future Improvements

* Implement **data augmentation** to improve model generalization
* Try using **Convolutional Neural Networks (CNN)** for better accuracy
* Hyperparameter tuning using **GridSearchCV**
* Implement **early stopping** during training to prevent overfitting
* Explore transfer learning with **pre-trained models** such as **ResNet**, **VGG**, etc.



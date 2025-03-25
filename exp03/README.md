# MNIST Handwritten Digit Classification

## Objective
This project aims to develop a deep learning model using TensorFlow to classify handwritten digits from the MNIST dataset. The model is a fully connected neural network (Multi-Layer Perceptron) that learns to recognize digits (0-9) based on their pixel values.

---

## Description of the Model
The model consists of:

- **Input Layer:** 784 neurons (each representing a pixel in the 28x28 image)
- **Hidden Layer 1:** 128 neurons with Sigmoid activation
- **Hidden Layer 2:** 64 neurons with Sigmoid activation
- **Output Layer:** 10 neurons (one for each digit class) with softmax activation
- **Loss Function:** Softmax cross-entropy
- **Optimizer:** Adam Optimizer
- **Batch Size:** 100
- **Epochs:** 20

---

## Description of the Code

### 1. Importing Required Libraries
- **TensorFlow:** For deep learning and neural network operations
- **NumPy:** For numerical computations
- **TensorFlow Datasets:** For loading the MNIST dataset

### 2. Loading & Preprocessing Data
- Load the MNIST dataset using `tf.keras.datasets.mnist.load_data()`
- Reshape images from 28x28 to 1D vectors (28x28 → 784)
- Normalize pixel values to a range of 0 to 1
- Convert labels to one-hot encoding using `np.eye()`

### 3. Building the Neural Network
- Define placeholders for inputs and outputs using `tf.compat.v1.placeholder`
- Initialize weights and biases using `tf.Variable`
- Perform feedforward propagation using matrix multiplications and sigmoid activation functions
- Compute the logits for the output layer

### 4. Training the Model
- Compute the loss using `tf.nn.softmax_cross_entropy_with_logits`
- Optimize using the Adam optimizer with a learning rate of 0.01
- Perform batch training for 20 epochs

### 5. Performance Evaluation
- Calculate the accuracy using the correct predictions compared to actual labels
- Evaluate the model on both training and test datasets

---

## Performance Evaluation

After training for **20 epochs**, the model achieves:

- **Training Accuracy:** ~98%
- **Test Accuracy:** ~97%

The model effectively classifies handwritten digits with high accuracy, demonstrating the power of deep learning for image recognition tasks.

---

## Comments on the Experiment

### **Strengths**
- Clear Neural Network Implementation
- Efficient Dataset Loading and Preprocessing
- Proper Use of TensorFlow Operations
- Effective Performance Metrics Evaluation

###  **Limitations & Suggested Improvements**
1. **TensorFlow 1.x Compatibility:**
    - The code uses `tf.compat.v1` and disables eager execution, limiting compatibility with modern TensorFlow 2.x.
    - Consider migrating to TensorFlow 2.x using `tf.keras.Sequential` for simpler code.

2. **Weight Initialization:**
    - Using `tf.random.truncated_normal` may result in poor training convergence.
    - Implement Xavier or He initialization for better performance.

3. **Lack of Regularization:**
    - No dropout or L2 regularization is used.
    - Add `tf.nn.dropout` or L2 regularization to prevent overfitting.

4. **Evaluation Method:**
    - Evaluating accuracy on the entire test set at once may cause memory issues.
    - Perform batch-wise evaluation for more efficient memory usage.

5. **Learning Rate Scheduling:**
    - A constant learning rate might not be optimal.
    - Implement learning rate decay using `tf.keras.optimizers.schedules`.

---

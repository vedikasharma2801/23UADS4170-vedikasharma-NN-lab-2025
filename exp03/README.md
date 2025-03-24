# MNIST Handwritten Digit Classification

## Objective
This project aims to develop a deep learning model using TensorFlow to classify handwritten digits from the MNIST dataset. The model is a fully connected neural network (Multi-Layer Perceptron) that learns to recognize digits (0-9) based on their pixel values.

---

## Description of the Model
The model consists of:

- **Input Layer:** 784 neurons (each representing a pixel in the 28x28 image)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 10 neurons (one for each digit class) with softmax activation
- **Loss Function:** Softmax cross-entropy
- **Optimizer:** Adam Optimizer
- **Batch Size:** 100
- **Epochs:** 10

---

## Description of the Code

### 1. Importing Required Libraries
- TensorFlow for deep learning
- NumPy for numerical operations
- TensorFlow Datasets for loading MNIST data

### 2. Loading & Preprocessing Data
- Load MNIST dataset using `tfds.load()`
- Normalize images (convert pixel values from 0-255 to 0-1)
- Flatten images to 1D vectors (28x28 → 784)
- One-hot encode labels (convert integer labels into binary vectors)
- Create batches of 100 images for efficient training

### 3. Building the Neural Network
- Define weight and bias tensors
- Implement forward propagation using matrix multiplications and activation functions
- Compute the loss using softmax cross-entropy
- Optimize the weights using the Adam optimizer

### 4. Training the Model
- Run training for 10 epochs
- Process batches dynamically using TensorFlow’s dataset API
- Compute the average loss per epoch

### 5. Performance Evaluation
- Evaluate model accuracy on the test set
- Measure how well the model generalizes to unseen data

---

## Performance Evaluation

After training for 10 epochs, the model achieves:

- **Training Accuracy:** ~98%
- **Test Accuracy:** ~97%

The model effectively classifies handwritten digits with high accuracy, demonstrating the power of deep learning for image recognition tasks.

---

## Comments on the Experiment

### Strengths
- Basic Neural Network Implementation
- Dataset Handling
- Normalization and One-Hot Encoding
- Optimization and Loss Function
- Evaluation:
The implementation of accuracy calculation using TensorFlow operations is well-executed, providing reliable performance metrics.

### Limitations
-**Use of TensorFlow 1.x:**
The code uses tf.compat.v1 and tf.disable_v2_behavior(), which limits compatibility with modern TensorFlow versions. Migrating to TensorFlow 2.x would ensure long-term support and more efficient development.

-**Random Initialization of Weights and Biases:**
tf.random_normal may not be the best choice for initializing weights. Using techniques like Xavier or He initialization could improve training stability and convergence.

-**Lack of Dropout or Regularization:**
No dropout or L2 regularization is applied, which may lead to overfitting. Implementing these techniques can enhance generalization.

-**Limited Error Handling:**
The use of tf.errors.OutOfRangeError for catching the end of the dataset is not optimal. Using for batch_x, batch_y in train_data: would be more Pythonic.

-**Evaluation on Entire Test Set at Once:**
Evaluating accuracy using the entire test set might lead to memory issues. It’s recommended to compute accuracy in batches.

-**Lack of Learning Rate Scheduling:**
Implementing a learning rate scheduler using tf.keras.optimizers.schedules can significantly improve training convergence.

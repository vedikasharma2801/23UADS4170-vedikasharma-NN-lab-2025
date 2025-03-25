## OBJECTIVE:
To implement a Multi-Layer Perceptron (MLP) using NumPy and demonstrate its ability to learn the XOR Boolean function. XOR is a non-linearly separable problem that requires at least one hidden layer to be correctly classified.

---

## Description of the Model:

A Multi-Layer Perceptron (MLP) is a type of artificial neural network consisting of an input layer, one or more hidden layers, and an output layer. It uses non-linear activation functions to learn complex patterns in data. For this task, we implement an MLP with:

- **Input layer:** Two neurons representing the binary inputs.
- **Hidden layer:** Four neurons implementing AND, OR, and NOR functions.
- **Output layer:** A single neuron combining hidden outputs to compute XOR.
- **Activation Function:** Uses a step function to determine output.
- **Forward Propagation:** Computes output at each layer using dot products and activation functions.

---

## Description of Code:

- **1.Perceptron Function:** Uses a step activation function to determine the neuron’s output.
- **2.Hidden Layer Processing:** Implements four perceptrons with pre-defined weights and biases to mimic logic functions.
- **3.Output Layer Processing:** Combines hidden layer outputs using another perceptron to compute XOR.
- **4.Testing:** The model is tested on all XOR truth table inputs and prints the final results.

---

## Performance Evaluation:

- **Accuracy:** The model correctly classifies XOR inputs, proving that the perceptron-based MLP is capable of solving the problem.
- **Loss Reduction:** Since a step function is used, loss is not explicitly calculated, but correct predictions indicate effective learning.
- **Confusion Matrix (Optional):** A table showing correct and incorrect classifications could be implemented.
- **Graphical Representation (Optional):** A plot of decision boundaries could help visualize how the perceptron separates classes.

---

## MY COMMENTS:

### (a) Limitations:
- The model is manually designed with fixed weights and biases, which does not generalize well to other problems.
- The perceptron’s step function limits gradient-based learning methods.


### (b) Scope for Improvement:
- Implement a trainable version using backpropagation to adjust weights.
- Use sigmoid or ReLU for hidden layers to enable smooth learning.
- Extend the model to classify multi-class logic functions beyond XOR.
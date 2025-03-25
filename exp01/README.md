## Objective:
To implement the Perceptron Learning Algorithm using NumPy in Python and evaluate its performance on NAND and XOR truth tables. The goal is to understand the capability of a single-layer perceptron in solving linearly separable and non-linearly separable problems.

---

## Description of the Model:
A perceptron is a simple type of artificial neural network that can classify data into two categories based on a weighted sum of inputs. The learning process involves adjusting the weights based on errors until the model converges. Key components include:

- **Step Activation Function:** Determines output as 0 or 1 based on thresholding.
- **Weight and Bias Update Rule:** Adjusts based on errors using a learning rate.
- **Training Process:** Iteratively updates weights until convergence or maximum epochs.
- **Prediction Function:** Applies learned weights to new data.

---

## Description of Code:

### 1.Step Function: 
A simple threshold function that outputs 1 if the input is >= 0, otherwise 0.

### 2.Perceptron Learning Algorithm:
- Initializes weights and bias to zero.
- Iterates through data for a fixed number of epochs.
- Computes weighted sum and applies step function.
- Updates weights and bias based on errors.
- Stops early if no errors occur.

### 3.Prediction Function:
- Computes the output for new inputs using trained weights.

### 4.Training and Evaluation:
- Trains the perceptron on NAND and XOR datasets.
- Prints learned weights and predictions.

---

## Performance Evaluation:

- **NAND Gate:** A single-layer perceptron successfully learns to model the NAND function since it is linearly separable.
- **XOR Gate:** The perceptron fails to learn XOR because XOR is not linearly separable, requiring at least a multi-layer perceptron.
- **Accuracy:** Perfect accuracy on NAND but fails on XOR.
- **Confusion Matrix (Optional):** Can be computed to visualize misclassifications.
- **Graphical Representation (Optional):** Decision boundary plots could illustrate separability.

---

## My Comments:

### (a)Limitations:
- A single perceptron cannot solve non-linearly separable problems like XOR.
- The step function restricts gradient-based optimization.

### (b)Scope for Improvement:
- Use a multi-layer perceptron (MLP) with a hidden layer to solve XOR.
- Replace step function with sigmoid or ReLU to allow gradient-based learning.
- Implement error visualization through loss curves or confusion matrices.
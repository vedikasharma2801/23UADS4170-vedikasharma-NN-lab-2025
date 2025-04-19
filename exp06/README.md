# 📈 Time Series Prediction Using RNN (PyTorch)

## 🎯 Objective
To train and evaluate a Recurrent Neural Network (RNN) using the PyTorch library to predict the next value in a sample time series dataset (e.g., sine wave).

---

## 🧠 Description of the Model
A simple RNN-based neural network is designed using PyTorch. The model takes a window of time steps as input and learns to predict the next time step in a sequence. A sine wave is used as the dataset to simulate time series behavior.

---

## 📄 Description of Code

### 1. Data Generation
- A sine wave is generated using `numpy` and scaled between 0 and 1 using `MinMaxScaler` from `sklearn`.

### 2. Model Architecture
- `RNNPredictor`: A basic RNN model with:
  - One RNN layer
  - One fully connected layer

### 3. Training
- Loss function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Input: sequences of fixed window size
- Output: predicted next time step

### 4. Evaluation
- Test MSE is printed after evaluation
- Training loss is plotted over epochs
- A prediction vs actual values graph is plotted

---

## 📊 Performance Evaluation

- **Loss Curve**: Visual representation of how training loss (MSE) decreases over epochs.
- **Test MSE**: Numerical evaluation of model performance on test data.
- **Prediction Plot**: Shows how closely the predicted values match the actual sine wave.

---

## My Comments

###  Limitations
- Simple RNNs suffer from vanishing gradient problems for long sequences.
- The model performs well on clean sine waves but may not generalize to noisy or real-world data.

###  Scope of Improvement
- Replace RNN with **LSTM** or **GRU** to better handle long-term dependencies.
- Tune hyperparameters such as hidden size, number of layers, and learning rate.
- Add **early stopping**, **cross-validation**, or **batch normalization** for better generalization.




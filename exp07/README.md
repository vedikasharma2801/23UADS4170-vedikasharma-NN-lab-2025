# 🧠 Medical Image Classification using Transfer Learning (ResNet-18)

## 🎯 Objective
The objective of this project is to develop a deep learning model capable of classifying medical images into different categories using a transfer learning approach with a pretrained ResNet-18 model.

---

## 📚 Description of Model

This project utilizes the **ResNet-18** architecture, a popular convolutional neural network pre-trained on the **ImageNet** dataset. 

- The final fully connected (FC) layer is replaced to match the number of classes in the **Medical MNIST** dataset.  
- By using **transfer learning**, the model leverages learned features from general image datasets and fine-tunes them for the medical image classification task.

---

## 💻 Description of Code

### 🛠️ Libraries Used:
- PyTorch
- Torchvision
- Matplotlib

### 📂 Dataset Handling:
- Images are loaded from a directory structure with `train/` and `val/` subfolders.
- Data augmentation techniques (e.g., random horizontal flip) and normalization are applied to improve generalization.

### 🧠 Model Definition:
- A **pretrained ResNet-18** model is loaded.
- The final FC layer is **modified** to output predictions based on the number of medical classes in the dataset.

### 🔁 Training Loop:
- Model is trained for **10 epochs**.
- Tracks **training and validation loss and accuracy**.
- The **best-performing model** on the validation set is saved for evaluation.

### 📊 Visualization:
- A **loss curve** is plotted at the end of training to visualize training progress.

---

## 📈 Performance Evaluation

### 🧪 Training & Validation Metrics:
- Accuracy and loss are printed for both training and validation phases across epochs.
- Best model is selected based on **highest validation accuracy**.

### 📉 Visualization:
- A **loss curve** is generated to track the decrease in loss over training epochs.
- Actual performance metrics (accuracy, loss) may vary depending on dataset quality and number of samples.

---

##  My Comments

### Suggestions for Future Improvement:
- Add **test set evaluation**
- Experiment with deeper networks (e.g., **ResNet-50**)
- Implement **learning rate scheduling** or **early stopping**


## Model Limitations

- If the dataset has **fewer images** or **uneven class distribution**, the model might get confused.
- ResNet-18 is **not very deep**; while effective, better-performing models exist.
- **No preprocessing or filtering** of bad or wrong images was implemented.

---


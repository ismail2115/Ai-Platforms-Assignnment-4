# 📌 **Fashion MNIST — Deep CNN Classification (Keras & PyTorch)**

This project implements a **deep Convolutional Neural Network (CNN)** trained on the **Fashion-MNIST** dataset using **two different frameworks**:

* **TensorFlow / Keras**
* **PyTorch**

Both implementations follow a well-structured pipeline:

1. Dataset loading
2. Preprocessing
3. Building a deep CNN
4. Training
5. Evaluation
6. Visualization of accuracy/loss curves
7. GPU support

---

## 📂 **Project Structure**

```
📁 Fashion-MNIST-CNN
│
├── keras_fashion_mnist_cnn.py      # Deep CNN using Keras
├── pytorch_fashion_mnist_cnn.py    # Deep CNN using PyTorch
└── README.md                       # Documentation
```

---

# 👗 **Fashion MNIST Dataset**

Fashion-MNIST contains **70,000 grayscale images (28×28)** across **10 categories**:

| Label | Class       |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

# 🧠 **Model Architecture Overview**

Both implementations use a **deep CNN** with:

### ✔ 3 Convolution Blocks:

* Conv2D → BatchNorm → ReLU
* MaxPooling
* Dropout

### ✔ Dense Classifier:

* Flatten
* Dense 256 → ReLU
* Dense 10 → Softmax / Logits

### ✔ Optimizer:

* **Adam** (lr = 0.001)

### ✔ Loss:

* **Sparse Categorical Crossentropy** (Keras)
* **CrossEntropyLoss** (PyTorch)

---

# 🚀 **How to Run**

## 📌 1. Install Dependencies

### **Keras version**

```bash
pip install tensorflow matplotlib numpy
```

### **PyTorch version**

```bash
pip install torch torchvision matplotlib numpy
```

---

## 📌 2. Run Keras Model

```bash
python keras_fashion_mnist_cnn.py
```

---

## 📌 3. Run PyTorch Model

```bash
python pytorch_fashion_mnist_cnn.py
```

---

# 📉 **Training Curves**

Both scripts automatically generate and save:

* `training_accuracy.png`
* `training_loss.png`

These charts help visualize overfitting/underfitting.

---

# 📊 **Expected Results**

| Framework   | Test Accuracy |
| ----------- | ------------- |
| **Keras**   | ~92–93%       |
| **PyTorch** | ~92–93%       |

(Your results may vary slightly depending on hardware.)

---

# ⚙️ GPU Support

Both implementations automatically detect a GPU:

### **Keras**

```python
tf.config.list_physical_devices("GPU")
```

### **PyTorch**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

# 🧪 **Evaluation Metrics**

Both scripts compute:

* Test accuracy
* Classification report
* Confusion matrix

---

# 📬 **Notes**

* Both implementations are written to be **as similar as possible** for easy comparison.
* This project is great for learning **deep learning**, **CNNs**, or comparing **Keras vs PyTorch** workflows.

---

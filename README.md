<h1 align="center">🧠 Handwritten Digit Classifier | TensorFlow + Keras</h1>

<p align="center">
  A clean and powerful deep learning project to classify handwritten digits using the MNIST dataset.
</p>


## 📌 Project Highlights

✅ Built a neural network using Keras  
✅ Performed extensive hyperparameter tuning (72 combinations)  
✅ Visualized accuracy trends across epochs, layers, activations, optimizers  
✅ Identified and saved the best performing model

---

## 🗂️ Repository Structure

| File / Folder                  | Description |
|-------------------------------|-------------|
| `base_model.ipynb`            | Base model for MNIST classification (Activity 1) |
| `hyperparameter_tuning.ipynb` | Full tuning of model parameters (Activity 2) |
| `README.md`                   | You're here! |

---

## 🎯 Activities Explained

### 🧪 Activity 1 – Base Model

**Goal:** Build and train a simple neural network to classify handwritten digits.

- Loaded and visualized MNIST dataset
- Normalized pixel values
- Built a 2-layer dense neural network with `relu` and `softmax`
- Achieved ~97.5% test accuracy

### 🔁 Activity 2 – Hyperparameter Tuning

**Goal:** Explore 72 combinations of hyperparameters including:
- Epochs: 5, 10, 20
- Layers: 1, 2
- Neurons: 64, 128, 256
- Activation: `relu`, `tanh`
- Optimizer: `adam`, `sgd`

🧠 Visualized insights across all combinations  
🏆 Best model: 2 layers, 128 neurons, `relu`, `adam`, 20 epochs — **98.41% accuracy**

---

## 📈 Top 10 Model Configurations

| # | Epochs | Layers | Neurons | Activation | Optimizer | Accuracy (%) |
|--:|--------|--------|---------|------------|-----------|---------------|
| 1 | 20     | 2      | 128     | relu       | adam      | **98.41**     |
| 2 | 20     | 2      | 256     | relu       | adam      | 98.35         |
| 3 | 10     | 2      | 128     | relu       | adam      | 98.12         |
| 4 | 20     | 2      | 64      | relu       | adam      | 98.07         |
| 5 | 10     | 2      | 256     | relu       | adam      | 98.01         |
| 6 | 10     | 1      | 256     | relu       | adam      | 97.95         |
| 7 | 20     | 1      | 128     | relu       | adam      | 97.93         |
| 8 | 20     | 2      | 256     | tanh       | adam      | 97.89         |
| 9 | 10     | 2      | 128     | tanh       | adam      | 97.82         |
|10 | 10     | 1      | 128     | relu       | adam      | 97.80         |

---

## 📊 Key Observations

- `relu` outperformed `tanh` in most setups.
- `adam` optimizer gave superior results over `sgd`.
- Best performance achieved with deeper networks (2 layers) and moderate neurons (128–256).
- Accuracy gains slowed beyond 10–20 epochs.
- Best model saved as `best_model.h5`

## 🧠 Tech Stack
1. Python

2. TensorFlow / Keras

3. NumPy, Pandas

4. Matplotlib, Seaborn

5. Jupyter / Colab

6. MNIST Dataset



## 🚀 How to Run

    # Clone the repo
    git clone https://github.com/bhagya-prog/Handwritten-Digit-Classifier.git
    cd Handwritten-Digit-Classifier

    # Open notebooks in Jupyter or Google Colab 



## 📄 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code.

## 🙌 Author
Made with ❤️ by Bhagya Vardhan <br>
If you find this project helpful, feel free to ⭐ the repo!
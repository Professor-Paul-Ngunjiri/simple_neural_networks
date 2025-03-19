# Neural Network Training

This project implements a simple **feedforward neural network** using **Python** and **NumPy**. The network is trained using **backpropagation** and visualizes the **error reduction over time** using **Matplotlib**.

---

## 📌 Project Overview

This neural network consists of:
✅ **An input layer** with two input values
✅ **A hidden layer** with two neurons
✅ **An output layer** with one neuron
✅ **Sigmoid activation function** for non-linearity
✅ **Backpropagation** for training and adjusting weights
✅ **Error visualization** using Matplotlib

The training process updates weights over multiple epochs to minimize error and improve prediction accuracy.

---

## 🛠️ Technologies Used

This project is built using:
- **Python** (Programming Language)
- **NumPy** (For numerical computations)
- **Matplotlib** (For visualizing training progress)

---

## 🔧 Installation Guide

To run this project on your local machine, follow these steps:

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Professor-Paul-Ngunjiri/simple_neural_networks.git
cd neural-network-training
```

### 2️⃣ Install Dependencies
```sh
pip install numpy matplotlib
```

### 3️⃣ Run the Script
```sh
python neural_network_backprop_simulation.py
```

---

## 📝 Code Explanation

The script implements a basic **feedforward neural network** with **one hidden layer** and trains it using **backpropagation**.

### 🔹 Core Functions

#### Sigmoid Activation Function
```python
# Sigmoid function to introduce non-linearity
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

#### Derivative of Sigmoid (Used for Backpropagation)
```python
# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)
```

### 🔹 Training Process
The network undergoes multiple **epochs** (iterations) where:
1. It **calculates weighted inputs** to the hidden and output layers.
2. It **applies the sigmoid activation function** to introduce non-linearity.
3. It **computes the error** (difference between predicted and actual output).
4. It **adjusts weights** using **backpropagation** to minimize the error.

```python
# Compute Error
target = 1
error = target - final_output

# Adjust weights using backpropagation
w1 += learning_rate * delta_output * hidden_output_1
w2 += learning_rate * delta_output * hidden_output_2
```

---

## 📊 Expected Output

During training, the network prints output showing:
- Neuron inputs & outputs
- Error reduction per epoch

```
Epoch 1/100
Hidden Neuron 1 Input: 0.32, Output: 0.58
Hidden Neuron 2 Input: 0.45, Output: 0.61
Final Neuron Input: 0.52, Output: 0.63
Error: 0.37
```

---

## 📈 Training Progress Visualization

The **error reduction over time** is plotted using Matplotlib:

```python
plt.plot(range(epochs), errors, label='Error Reduction', color='red')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Neural Network Learning Progress')
plt.legend()
plt.grid()
plt.show()
```

### Example Graph Output:
![Error Reduction Graph](error_graph.png)

This graph shows how the **error decreases** as the neural network learns.

---

## 📚 Further Improvements

Potential enhancements for this project include:
✅ **Increasing hidden layers** to make the model more complex
✅ **Using different activation functions** like ReLU or Tanh
✅ **Implementing a larger dataset** for better generalization
✅ **Introducing learning rate decay** for optimized training

---

## 🔗 Useful Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Backpropagation Algorithm Explained](https://en.wikipedia.org/wiki/Backpropagation)

---

## 💡 About the Author

👤 **Professor-Paul-Ngunjiri**  
🔗 [GitHub](https://github.com/Professor-Paul-Ngunjiri)
📧 ngunjiripaul485@gmail.com  

---

## 📜 License
This project is licensed under the **GNU GENERAL PUBLIC LICENSE**. Feel free to use and modify the code as needed!

```md
GNU GENERAL PUBLIC LICENSE
Copyright (c) 2025
```


# CNN-MNIST

This repository contains a CNN model trained to recognise handwritten digits from the MNIST dataset.

**Performance metrics (Testing dataset):**

accuracy  : 99.67%
precision : 99.67%
recall    : 99.67%

**_./main.py_**: The model implementation, training, testing, and checkpoint storage.

**_./checkpoint_**: Saves the model weights to mps and cpu.

**_./app.py_**: A Streamlit app for handwritten digit recognition from user input with the model trained in _./main.py_.
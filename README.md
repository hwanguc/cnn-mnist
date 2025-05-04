# CNN-MNIST

This repository contains a **full-stack**, **end-to-end** deployment of CNN model trained to recognise handwritten digits from the MNIST dataset.

Access the web app at: 🔗 [Live Deployment](https://mnist-decoder.onrender.com)

**Performance metrics (Testing dataset):**

accuracy  : 99.67%
precision : 99.67%
recall    : 99.67%

**_./main.py_**: The model implementation, training, testing, and checkpoint storage.

**_./checkpoint_**: Saves the model weights to mps and cpu.

**_./app.py_**: A Streamlit app for handwritten digit recognition from user input with the model trained in _./main.py_.

**_./requirements.txt_**: Python packages required to run the web app locally.
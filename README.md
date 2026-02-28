NEURAL NETWORK FROM SCRATCH (NUMPY ONLY)

This project implements a fully connected neural network from scratch using only NumPy, without deep learning frameworks.
The network is trained on the MNIST dataset for handwritten digit classification.


# PROBLEM: MNIST CLASSIFICATION
    - Input images: 28 x 28 pixels
    - Flattened input size: 784 features
    - Output classes: 10 (digits 0–9)
    - Task type: Multiclass classification

# NEURAL NETWORK ARCHITECTURE
# The network consists of three layers:
    Input Layer:
    - 784 nodes (one per pixel)
    - No trainable parameters
    Hidden Layer
    - 10 neurons
    - Activation function: ReLU
    Output Layer
    - 10 neurons (one per class)
    - Activation function: Softmax

# FORWARD PROPAGATION
    Let:
        >  A[0] = X (input data)
        >  W[l] = weights of layer l
        >  b[l] = bias of layer l

    Hidden Layer:
        >  Z[1] = W[1] dot A[0] + b[1]
        >  A[1] = ReLU(Z[1])

    Output Layer:
        >  Z[2] = W[2] dot A[1] + b[2]
        >  A[2] = softmax(Z[2])

# ACTIVATION FUNCTIONS
    ReLU (Rectified Linear Unit):
        - ReLU(x) = x if x > 0
        - ReLU(x) = 0 if x <= 0
        - ReLU introduces non-linearity into the model.

    Softmax:
        - softmax(zi) = exp(zi) / sum(exp(zj)) for j = 1 to K
        - Properties: Each output value is between 0 and 1 and all outputs sum to 1
        - Used for multiclass classification
        - Sigmoid is typically used for binary classification.
        - Softmax is used for multiclass classification.

# BACKPROPAGATION
    Backpropagation computes gradients of the loss function with respect to weights and biases.
    Let:
    m = number of training examples
    Y = true labels (one-hot encoded)
    A[2] = predicted probabilities
    OUTPUT LAYER:
        >  dZ[2] = A[2] - Y
        >  dW[2] = (1/m) * dZ[2] dot A[1]^T
        >  db[2] = (1/m) * sum(dZ[2])
    HIDDEN LAYER:
        >  dZ[1] = (W[2]^T dot dZ[2]) * ReLU'(Z[1])
        >  dW[1] = (1/m) * dZ[1] dot A[0]^T
        >  db[1] = (1/m) * sum(dZ[1])

# PARAMETER UPDATE (GRADIENT DESCENT)
    >  W[1] = W[1] - alpha * dW[1]
    >  b[1] = b[1] - alpha * db[1]
    >  W[2] = W[2] - alpha * dW[2]
    >  b[2] = b[2] - alpha * db[2]
    Where:
    alpha = learning rate (user-defined hyperparameter)

# SUMMARY
    - This project demonstrates:
    - Forward propagation
    - ReLU and Softmax implementation
    - Backpropagation using matrix operations
    - Gradient descent optimization
    - Multiclass classification with one-hot encoding

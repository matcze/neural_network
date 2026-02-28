Neural Network from Scratch (NumPy Only)

This project implements a fully connected neural network from scratch using NumPy, without relying on any deep learning frameworks. The network is trained on the MNIST dataset for handwritten digit classification.

Problem

MNIST Classification

Input images: 28 × 28 pixels

Flattened input size: 784 features

Output classes: 10 (digits 0–9)

Task type: Multiclass classification

Neural Network Architecture

The network consists of three layers:

Input Layer

784 nodes (one per pixel)

No trainable parameters

Hidden Layer

10 neurons

Activation function: ReLU

Output Layer

10 neurons (one per class)

Activation function: Softmax

Forward Propagation

Let:

𝐴
[
0
]
=
𝑋
A[0]=X (input data)

𝑊
[
𝑙
]
W[l] = weights of layer 
𝑙
l

𝑏
[
𝑙
]
b[l] = bias of layer 
𝑙
l

Hidden Layer:

𝑍
[
1
]
=
𝑊
[
1
]
⋅
𝐴
[
0
]
+
𝑏
[
1
]
Z[1]=W[1]⋅A[0]+b[1]
𝐴
[
1
]
=
𝑅
𝑒
𝐿
𝑈
(
𝑍
[
1
]
)
A[1]=ReLU(Z[1])

Output Layer:

𝑍
[
2
]
=
𝑊
[
2
]
⋅
𝐴
[
1
]
+
𝑏
[
2
]
Z[2]=W[2]⋅A[1]+b[2]
𝐴
[
2
]
=
𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑍
[
2
]
)
A[2]=softmax(Z[2])
Activation Functions

ReLU (Rectified Linear Unit)

𝑅
𝑒
𝐿
𝑈
(
𝑥
)
=
{
𝑥
	
if 
𝑥
>
0


0
	
if 
𝑥
≤
0
ReLU(x)={
x
0
	​

if x>0
if x≤0
	​


Introduces non-linearity into the model

Softmax

𝑠
𝑜
𝑓
𝑡
𝑚
𝑎
𝑥
(
𝑧
𝑖
)
=
𝑒
𝑧
𝑖
∑
𝑗
=
1
𝐾
𝑒
𝑧
𝑗
softmax(z
i
	​

)=
∑
j=1
K
	​

e
z
j
	​

e
z
i
	​

	​


Each output value is between 0 and 1

Outputs sum to 1

Used for multiclass classification

Note: Sigmoid is typically used for binary classification, while softmax is preferred for multiclass problems.

Backpropagation

Backpropagation computes gradients of the loss with respect to weights and biases.

Let:

𝑚
m = number of training examples

𝑌
Y = true labels (one-hot encoded)

𝐴
[
2
]
A[2] = predicted probabilities

Output Layer:

𝑑
𝑍
[
2
]
=
𝐴
[
2
]
−
𝑌
dZ[2]=A[2]−Y
𝑑
𝑊
[
2
]
=
1
𝑚
𝑑
𝑍
[
2
]
⋅
𝐴
[
1
]
𝑇
dW[2]=
m
1
	​

dZ[2]⋅A[1]
T
𝑑
𝑏
[
2
]
=
1
𝑚
∑
𝑑
𝑍
[
2
]
db[2]=
m
1
	​

∑dZ[2]

Hidden Layer:

𝑑
𝑍
[
1
]
=
(
𝑊
[
2
]
𝑇
⋅
𝑑
𝑍
[
2
]
)
∗
𝑅
𝑒
𝐿
𝑈
′
(
𝑍
[
1
]
)
dZ[1]=(W[2]
T
⋅dZ[2])∗ReLU
′
(Z[1])
𝑑
𝑊
[
1
]
=
1
𝑚
𝑑
𝑍
[
1
]
⋅
𝐴
[
0
]
𝑇
dW[1]=
m
1
	​

dZ[1]⋅A[0]
T
𝑑
𝑏
[
1
]
=
1
𝑚
∑
𝑑
𝑍
[
1
]
db[1]=
m
1
	​

∑dZ[1]
Parameter Update (Gradient Descent)
𝑊
[
1
]
=
𝑊
[
1
]
−
𝛼
⋅
𝑑
𝑊
[
1
]
W[1]=W[1]−α⋅dW[1]
𝑏
[
1
]
=
𝑏
[
1
]
−
𝛼
⋅
𝑑
𝑏
[
1
]
b[1]=b[1]−α⋅db[1]
𝑊
[
2
]
=
𝑊
[
2
]
−
𝛼
⋅
𝑑
𝑊
[
2
]
W[2]=W[2]−α⋅dW[2]
𝑏
[
2
]
=
𝑏
[
2
]
−
𝛼
⋅
𝑑
𝑏
[
2
]
b[2]=b[2]−α⋅db[2]

Where 
𝛼
α is the learning rate, a user-defined hyperparameter.

Summary

This project demonstrates:

Forward propagation using NumPy

Implementation of ReLU and Softmax activations

Backpropagation using matrix operations

Gradient descent optimization

Multiclass classification with one-hot encoding
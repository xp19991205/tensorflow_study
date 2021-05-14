import tensorflow as tf
import numpy as np
rng = np.random
# Parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50
# Training Data.训练数据
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# Weight and Bias, initialized randomly. 随机初始化权重和偏置
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Linear regression (Wx + b). 返回前向传播的值
def linear_regression(x):
    return W * x + b

# Mean square error.
def mean_square(y_pred, y_true): #计算误差值
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent Optimizer.随机梯度下降法
optimizer = tf.optimizers.SGD(learning_rate)


def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    #把计算包裹在里面，以实现自动微分
    with tf.GradientTape() as g: #将GradientTape作为g来引用
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # Compute gradients. 计算梯度
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients. 根据梯度下降更新w 和b
    optimizer.apply_gradients(zip(gradients, [W, b]))
for step in range(1, training_steps + 1):
        # Run the optimization to update W and b values.
    run_optimization()
    if step % display_step == 0:
            pred = linear_regression(X)
            loss = mean_square(pred, Y)
            print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))

import matplotlib.pyplot as plt
# Graphic display
plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()
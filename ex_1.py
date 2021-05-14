import tensorflow as tf
import numpy as np

data_x = np.random.normal(size=[1000, 1])
noise = np.random.normal(size=[1000, 1]) * 0.2
data_y = data_x * 3.4 + 2. + noise


class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.w = tf.Variable(0.1, dtype=tf.float32)
        self.b = tf.Variable(0.1, dtype=tf.float32)

    def call(self, x):
        return self.w * x + self.b


model = Model()
var_list = [model.w, model.b]
opt = tf.optimizers.SGD(0.1) #gradient decent 梯度下降优化器，下降速度为0.1

for t in range(300): #迭代100次
    with tf.GradientTape() as tape:
        y_ = model.call(data_x) #前向传播，计算模型的输出值
        loss = tf.reduce_mean(tf.square(data_y - y_)) #计算损失函数

    grad = tape.gradient(loss, var_list) #计算当前梯度
    opt.apply_gradients(zip(grad, var_list)) #zip函数的用法zip([1,2,3],[2,3,4]) = [[1,2,3],[2,3,4]]
    if t % 10 == 0: #每10次打印一次
        print("loss={:.2f} | w={:.2f} | b={:.2f}".format(
            loss, model.w.numpy(), model.b.numpy())
        )
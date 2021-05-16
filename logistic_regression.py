#这个程序的主要思路是，把28*28的图片，拆成1个1个像素点(0.256色阶)，然后把这个值归一化倒0-1
#然后使用了交叉熵作为代价函数，将这个代价函数优化至最小，然后进行输出
import tensorflow as tf
import numpy as np
# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits 0-9数字 共10类
num_features = 784 # 28*28 总共有这么多像素点

# Training parameters.
learning_rate = 0.01 #学习率
training_steps = 1000 #迭代次数
batch_size = 256 #一次训练所抓取的数据样本数量；
display_step = 50 #每50步展示一次

# Prepare MNIST data. 加载数据
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()#这里的训练集有60000张图片，测试集有10000张图片
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28). 把二维图片处理成28*28个特征点(对应像素)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features]) #这里的-1代表模糊控制，不知道多少行 10000*784
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1) #加载数据，打乱shuffle，预处理
# 将数据打乱，数值越大，混乱程度越大 batch 按照顺序取出batch_size行数据，最后一次输出可能小于batch_size  repeat:数据集重复了指定次数
# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias") #每个像素点都有一个与之对应的W b

# Logistic regression (Wx + b).
def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b) #输出套了一个softmax:用于预测最大概率的类

# Cross-Entropy loss function. 交叉熵函数，用于计算代价
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)#生成onehot类型的张量例如 [1，0，0，0.。。] 举例来说这里就是代表是哪类，例如1，对应的就是[0,1,0,0,0....]
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.) #这个函数限制y的值域，小于1e-9置为1e-9,大于1，置为1
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1)) #-y*log(p)求和，熵

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).取的是
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))#tf.cat数据类型转换 tf.equal判断两个张量是否相等 tf.argmax返回tensor中最大值的下标
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #计算平均正确率
#tf.argmax(y_pred, 1)参数取1代表求行元素最大值的索引（下标），如果取0，则代表求列元素最大值的下标

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)


# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y) #越确定，熵越小

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))
##调整 w b 使得损失函数（熵最小化）

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_regression(batch_x)# (256,10)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set.
pred = logistic_regression(x_test)
print(pred)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# Visualize predictions.
import matplotlib.pyplot as plt

# Predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

# Display image and model prediction.
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
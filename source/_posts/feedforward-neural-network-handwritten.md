---
title: 前馈神经网络实践
date: 2017-08-08 14:07:05
tags: [神经网络,机器学习,梯度下降,反向传播]
categories: 深度学习
---
　　本文将介绍使用Python手写“前馈神经网络”代码的具体步骤和细节。前馈神经网络采用随机梯度下降算法进行学习，代价函数的梯度计算方法使用的是反向传播算法。具体代码可参考[神经网络和深度学习教程][0]。
<!--more-->
# 数据集的加载
　　本文使用的数据集来自MNIST。MNIST是一个手写数字的数据库，它提供了六万的训练集和一万的测试集。每个手写数字图片都已经被规范处理过，是一张放在中间部位的28px\*28px的灰度图。本文将训练集进一步划分成训练集和验证集，得到50000条训练集，10000条验证集，10000万条测试集。
## 细节代码
```python
mnist_loader.py：
def load_data():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)
```
　　上述代码cPickle.load(f)返回一个元组，包含训练集，验证集和测试集。训练集返回的是一个由两个元素构成的元组，第一个元素是包含50000条数据的numpy.ndarray，规格是(50000L, 784L)。即每条数据有784个值，代表了该手写数字的28\*28=784个像素点。第二个元素是numpy ndarray数组，规格的是(50000L,)，每个分量代表的是第一个元素中相对应位置的数据的数字分类。
　　可以发现该数据虽然很规整，但在神经网络中用起来不够方便，我们需要对数据进一步处理一下。
```python
mnist_loader.py：
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]#把数据处理成列向量形式
    training_results = [vectorized_result(y) for y in tr_d[1]]#把分类标记向量化
    training_data = zip(training_inputs, training_results)#每条数据都是2-tuples形式
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
```
　　上述处理后，训练集是一个包含50000条数据，每一条数据都是2-tuples(x,y)形式，x是一个784维度的numpy.ndarray，规格是(784L, 1L)，y是一个10维numpy.ndarray，规格是(10L, 1L)，代表的是相应数字分类的单位向量。验证集和测试集类似，只不过y是具体的数字分类，而不是向量。
## 具体使用
　　具体使用时，只要如下代码即可：
```python
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```

# 神经网络算法
## Network对象初始化
　　我们需要对神经网络结构进行初始化，具体而言就是对权重和偏置进行初始化。
```python
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
net = Network([784, 30, 10])
```
　　sizes参数代表的是神经网络的层数以及每层的神经元个数。例如[784,30,10]代表输入层有784个神经元，隐藏层有30个神经元，输出层有10个神经元。后面的例子都以该神经网络形状来阐述。
　　biases是神经元的偏置，输入层神经元不存在偏置，故从隐藏层开始，使用np.random.randn生成\\(N \sim (0,1)\\)的高斯分布。np.random.randn的参数代表维度数，例如(y,1)代表y行,1列的数组。因此biases是一个list，按照层的顺序进行存放，每个元素代表一层神经元的偏置的列向量(np.ndarray类型)。
　　而weights代表的是权重list。第一个元素代表从输入层到隐藏层的权重矩阵，矩阵的行是隐藏层(后一层)的神经元个数，列是输入层(前一层)的神经元个数。以此类推，按顺序存放。
## 前向传播算法
　　对于特定的输入，返回对应的输出的方法如下：
```python
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
net.feedforward([[5],[6]])#假设两个输入神经元数据
```
　　对于[784, 30, 10]结构的神经元，在第一次循环时，因为输入层到隐藏层的权重矩阵的规格是30\*784，则w规格为(30L,784L)；输入神经元a的规格是(784L,1L)。因此w.a矩阵相乘结果即(30L,784L)\*(784L,1L)，则为(30L,1L)，行数即隐藏层神经元的个数。
　　后面循环的过程中，直接将前一层的输出作为下一层的输入即可。
## 随机梯度下降算法
```python
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)#打乱数据
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]#等大小划分数据
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)#随机梯度下降
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
```
　　training_data每条数据都是一个(x,y)元组的列表，表示训练输入和其对应的期望输出。变量epochs代表迭代数量，mini_batch_size代表采样时的小批量数据的大小。eta是学习速率\\(\eta\\)。如果给出了可选参数test_data，那么程序会在每个训练器后评估神经网络，并打印出部分进展，这对于追踪进度很有用。
　　代码如下工作。在每个迭代期，首先随机地将训练数据打乱，然后将它分成多个适当大小的小批量数据。然后对于每一个mini_batch应用一次梯度下降。
　　注意上述迭代是在上一次迭代结果的基础上继续迭代的。每次迭代唯一不同的是划分的数据mini_batch不相同。因此结果应该是不断优化的。当然应该会存在一个阈值。
```python
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]#更新权重矩阵
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]#更新偏置矩阵
```
　　上述代码是用来更新权重和偏置。首先根据权重和偏置矩阵的规格对\\(\nabla b\\)和\\(\nabla w\\)梯度进行初始化。然后对每个min_batch使用后向传播算法计算梯度。根据:
$$\frac{\sum_{j=1}^m \nabla C_{X_j}}{m} \approx \frac{\sum_x \nabla C_x}{n}=\nabla C$$
　　这里的第二个求和符号是在整个训练集数据上进行的。交换两边得到：
$$\nabla C \approx \frac{1}{m}\sum_{j=1}^m \nabla C_{X_j}$$
　　即小批量数据上\\(\nabla C_{X_j}\\)的平均值大致相等于整个\\(\nabla C_{X}\\)的平均值。
　　因此：
$$w_k := w_k - \frac{\eta}{m} \frac{\partial C_{X_j}}{\partial w_k}$$
$$b_l := b_l - \frac{\eta}{m} \frac{\partial C_{X_j}}{\partial b_l}$$
　　对应代码是上述最后两句。
　　实际上，偏置的更新不需要独立出来，只需要在每层增加一个神经元\\(x_0=1\\),这样就可以将偏置当作权重来对待了，使得代码更加简洁。
　　另外，测试集上性能评估函数如下：
```python
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]#输出层有10个输出，数值最大的那个神经元的下标即为数字分类结果
        return sum(int(x == y) for (x, y) in test_results)#一共多少个相等
```
　　输出层有10个输出，根据sigmoid函数的图像，数值越大，代表可能性越高。因此数值最大的那个神经元的下标即为数字分类结果。
## 后向传播算法
　　最后我们来重点研究下后向传播算法如何快速求得参数的梯度。
　　注意，**本部分采用的代价函数是二次型的，下面所有的公式都是针对二次型代价函数而言**。之后我还会讨论交叉熵代价函数。
　　首先回顾一下，BP算法的四个重要公式：
![bp][3]
　　第一个公式BP1代表输出层的误差计算方法。BP2代表其他层的误差计算方法。BP3代表偏置的梯度求法。BP4代表权重的梯度求法。注意，\\(\odot\\)是Hadamard乘积，即相同位置上的数相乘。
```python
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)#未激活
            activation = sigmoid(z)#激活单元
            activations.append(activation)

        # backward pass 先计算输出层
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

def cost_derivative(self, output_activations, y):
    return (output_activations-y)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
```
　　这里的参数x,y代表一条数据，因此该方法一次只对一条数据进行后向传播来计算梯度。首先进行初始化，根据权重矩阵和偏置矩阵的规格进行初始化。activation代表当前激活单元，activations保存所有的激活单元，按层存放，每层使用一个数组来存储。zs保存未激活前的单元，同理按层存放，每层使用一个数组来存储。因此这里的运算都是矩阵运算。
　　接着从输出层开始，反向进行误差的计算。cost_derivative方法根据代价函数来求得对输出激活单元的导数。这里使用的是二次代价函数:
$$C(w,b)=\frac{1}{2n} \sum_{x} ||y(x)-a||^2$$
　　这里的n代表样本量，a代表输入为x时输出的向量(激活单元)，y(x)是数据的真实标记向量。
　　如果样本量为1，则：
$$\frac{\partial C}{\partial a}=-(y-a)=a-y$$
　　sigmoid_prime是对sigmoid函数求导。
　　我们想求输出层偏置的梯度\\(\frac{\partial C}{\partial b}\\)。首先分析下输出层偏置影响了哪些东西。令输出层偏置为\\(b\\)，输出层最终输出为\\(a\\)(激活了)，输出层激活前为\\(z\\)(前一层输出和权重线性组合后，未经过sigmoid函数处理)。
　　因为\\(z=wa'+b\\),其中\\(a'\\)是前一层的输出，因此\\(b\\)首先影响\\(z\\)。紧接着因为\\(a=sigmoid(z)\\),因此\\(z\\)影响\\(a\\)。而根据代价函数,\\(a\\)最终再影响代价函数。因此，根据链式求导法则：
$$\frac{\partial C}{\partial b} = \frac{\partial C}{\partial a} \* \frac{\partial a}{\partial z} \* \frac{\partial z}{\partial b} \\\\ 
=(a-y)\* [sigmoid(z)(1-sigmoid(z)] \* 1 \\\\ $$
　　上述式子可以逆着写，即从最后一个式子开始往前写。该结果和上述代码相符。
　　注意到，这里的乘法使用的是Hadamard乘积。向量的点乘np.dot是我们熟悉的矩阵运算。因此delta = self.cost_derivative(activations[-1], y)\*sigmoid_prime(zs[-1])是第一种情况，得到的delta规格也是10\*1。
　　紧接着我们想求是输出层权重的梯度\\(\frac{\partial C}{\partial w}\\)。同理分析，\\(w\\)首先影响\\(z\\),\\(z=wa'+b\\)。\\(z\\)再影响\\(a\\),\\(a=sigmoid(z)\\)。\\(a\\)最终影响代价函数，因此根据链式求导法则：
$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \* \frac{\partial a}{\partial z} \* \frac{\partial z}{\partial w} \\\\ 
=(a-y)\* [sigmoid(z)(1-sigmoid(z)] \* a' \\\\ $$
　　其中\\(a'\\)是前一层的输出。可以发现只要定义前一层\\(a_0=1\\)，则\\(w,b\\)的求导可以统一起来，不需要分开求。
　　注意delta的规格10\*1，也就是输出层10个神经元每个都有一个分量。activations[-2]的规格是30\*1，也就是前一层的神经元每个输出都有一个分量。activations[-2]转置后规格为1\*30,delta\*activations[-2].transpose()的结果矩阵的规格就是10\*30。也就是说对于输出层每一个神经元的误差，都需要分摊到连接该神经元的所有权重上。
　　接着看误差继续往前传播。
　　我们分析隐藏层的偏置\\(b'\\)的梯度。\\(b'\\)首先影响该层的输出\\(z'\\),\\(z'=w'a''+b'\\)；\\(z'\\)影响\\(a'=sigmoid(z')\\)；\\(a'\\)作为最后一层的输入影响z，\\(z=wa'+b\\)；\\(z\\)再影响\\(a=sigmoid(z)\\)，\\(a\\)最终影响代价函数\\(C\\)，根据链式法则：
$$\frac{\partial C}{\partial b'} = (\frac{\partial C}{\partial a} \* \frac{\partial a}{\partial z})
\* (\frac{\partial z}{\partial a'}) \* (\frac{\partial a'}{\partial z'}) \* (\frac{\partial z'}{\partial b'}) \\\\
=delta \* w \* sigmoid\\_prime(z') \* 1$$
　　可以看出delta是之前后一层求得的，\\(w\\)是后一层的权重，\\(z'\\)是该层的输出(未激活)。
　　和代码中的公式相符。可以看出权重矩阵w规格是10\*30, delta是10\*1，sigmoid_prime(z')是30\*1, 则np.dot(self.weights[-l+1].transpose(), delta) \* sp规格是30\*1，也就是隐藏层30个神经元每个都有一个新的delta分量。
　　\\(w'\\)梯度的分析类似。

# 运行结果
　　具体调用代码如下图所示：
![network][1]
　　这里使用的是3层神经网络，输入层有784个神经元，隐藏层30个神经元，输出层10个神经元，进行30次迭代，每个mini_batch有10个数据，学习率设置为3。
　　可以看出随着迭代次数的增加，性能基本上也都有稳步提升。
![network][2]


# 参考
[神经网络和深度学习入门][0]



[0]: http://neuralnetworksanddeeplearning.com/chap1.html
[1]: /picture/machine-learning/network_writting1.png
[2]: /picture/machine-learning/network_writting2.png
[3]: /picture/machine-learning/network_writting3.png
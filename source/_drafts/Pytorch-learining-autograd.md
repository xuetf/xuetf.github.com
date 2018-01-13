---
title: Pytorch learining (1)
date: 2018-01-04 14:58:11
tags: [Pytorch,学习笔记]
categories: 语言学习
layout: false
---

# Autograd Mechanics

## Excluding subgraphs from backward
有时我们会预训练好网络，例如固定卷积层作为图像特征提取器，用当前数据只训练全连接层，那么PyTorch如何在训练时只计算全连接层的梯度呢？ 此时可以使用每个变量的两个参数requires_grad和volatile来排除部分图节点，不进行反向传播计算梯度。
```python
>>> x = Variable(torch.randn(5, 5))#默认requires_grad为False
>>> y = Variable(torch.randn(5, 5))
>>> z = Variable(torch.randn(5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False #所有输入全部为False，输出才为False
>>> b = a + z
>>> b.requires_grad
True #有一个为True，输出就为True
```
当确定不再调用backward时，可以设置volatile=True，这意味着requires_grad=False，保证用最小的内存进行模型推断。volatile和requires_grad不同点在于flag标志传播方式。只要有一个输入叶子节点Volatile=True则输出变量的Volatile就为True，而requires_grad必须所有输入为False，输出变量的requires_grad才为False。
```python
>>> regular_input = Variable(torch.randn(1, 3, 227, 227))
>>> volatile_input = Variable(torch.randn(1, 3, 227, 227), volatile=True)
>>> model = torchvision.models.resnet18(pretrained=True)
>>> model(regular_input).requires_grad
True
>>> model(volatile_input).requires_grad
False
>>> model(volatile_input).volatile
True
>>> model(volatile_input).grad_fn is None
True
```
使用Volatile=True，那么就没必要改变模型的参数来进行推断，因为中间过程状态不会被保存。

## How autograd encodes the history
Autograd是一个反向自动微分系统。从概念上理解，autograd会构建一个有向无环图来记录所有的操作。这个图的叶子节点是输入变量，根节点是输出变量。从根节点跟踪到叶节点，可以使用链式法则自动计算处梯度。
从原理上理解，autograd表示的图的边是Funtion对象，可以被用来计算图节点变量的结果。当进行前向传播时，aotograd会进行相关的计算，同时会建立用于梯度计算的图。相当于每个变量是图上的节点，而每个变量的.grad_fn属性会加入图中，延生出一条边，连接下一个节点，进一步进行前向传播。Function相当于是连接这些节点的边。当前向传播计算完成后，会通过后向传播来计算梯度。
一个重要的点是，每次迭代步都会重新建图，这意味着我们可以使用任意的控制流，每次迭代都可以改变图的形式和大小，不需要在训练前就预先编码所有可能的路径。
## In-place operations on Variables
Autograd中使用原地操作比较困难，官方不建议使用。因为Autograd中缓存释放和复用做的十分高效，原地操作实际上不会降低多少内存的使用，另外原地重写变量的值需要重新计算梯度(这也是变量为啥不支持log_等原地操作的原因)，原地操作还需要重新实现计算图，需要修改和该变量相关的所有操作function，这通常会很棘手，因为如果有多个变量共享同一个内存地址(created by indexing or transposing)，那么相关的functions会在某个变量进行原地操作，同时存在其他变量指向该内存地址的时候报错。而如果不进行原地操作的话，只需要分配新的对象，并保持引用指向旧图即可。除非有很大的内存压力，否则不建议使用。
## In-place correctness checks
每个变量会保存一个版本号，会在每次操作被标记为脏的时候增加。当一个Function保存张量用于反向传播时，相应的版本号也会被保存。当访问self.saved_tensors，会进行版本号检验，当实际的版本号大于保存的版本号时会报错。

# Broadcasting semantics
Pytorch中很多操作支持numpy的广播机制。
广播的前提有2个：
- 1）是每个张量至少有一个维度；
- 2）尾部对齐，对应维度上的数要么相等，要么其中之一为1，要么其中之一不存在。
```python
>>> x=torch.FloatTensor(5,7,3)
>>> y=torch.FloatTensor(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

>>> x=torch.FloatTensor()
>>> y=torch.FloatTensor(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
>>> x=torch.FloatTensor(5,3,4,1)
>>> y=torch.FloatTensor(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(  3,1,1)
    # x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3
```
如果两个张量可以广播，那么得到的结果张量形状按照如下计算：
- 如果两个张量维度数不相同，那么在维度少的张量头部添加新维度且值为1，使得两个张量维度相同。然后按照下一条计算。
- 对于每个维度，选择维度size大的作为结果张量的维度size。
```python
# can line up trailing dimensions to make reading easier
>>> x=torch.FloatTensor(5,1,4,1)
>>> y=torch.FloatTensor(  3,1,1)
>>> (x+y).size()
torch.Size([5, 3, 4, 1])

# but not necessary:
>>> x=torch.FloatTensor(1)
>>> y=torch.FloatTensor(3,1,7)
>>> (x+y).size()
torch.Size([3, 1, 7])

>>> x=torch.FloatTensor(5,2,4,1)
>>> y=torch.FloatTensor(3,1,1)
>>> (x+y).size()
RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
```
另外对于原地操作，不允许改变要进行原地操作的张量的shape。
```python
>>> x=torch.FloatTensor(5,3,4,1)
>>> y=torch.FloatTensor(3,1,1)
>>> (x.add_(y)).size()
torch.Size([5, 3, 4, 1])

# but:
>>> x=torch.FloatTensor(1,3,1)
>>> y=torch.FloatTensor(3,1,7)
>>> (x.add_(y)).size()
RuntimeError: The expanded size of the tensor (1) must match the existing size (7) at non-singleton dimension 2.
```
版本兼容问题。对于1维数组，早期版本使用pointwise策略，新的版本使用广播策略。
```python
>>> torch.utils.backcompat.broadcast_warning.enabled=True
>>> torch.add(torch.ones(4,1), torch.ones(4))
__main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, and have the same number of elements.
Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 1-dimensional.
```
旧版计算结果为(4,1)，新版本使用广播策略，计算结果为(4,4).

# Extending PyTorch
本节学习如何扩展torch.nn和torch.autograd。
## Extending torch.autograd
为autograd添加新的操作需要为每一个操作实现一个Function的子类。回顾一下,Function是autograd用来计算激活值和梯度值，并进行历史操作记录的对象。每个function类需要实现2个方法，forward和backward。
- forward():用来执行操作的代码。可以传入任意的参数。Variable参数会在forward调用前转成Tensor，然后实际调用的时候再转回Variable并注册到图中。返回值可以是单个Tensor，也可以是tuple of Tensor。可以查下Function手册看下哪些有用的方法只允许在forward中调用。
- backward():梯度计算公式。输入Variable参数的个数要和输出一样多，每个参数代表相应的要输出的梯度值。如果某个输入参数不需要计算梯度值或输入参数的类型不是Variable,那么可以返回None.
```python
class LinearFunction(Function):
    # forward 和 backward 都得是 静态方法！！！！！
    @staticmethod
    # bias 是个可选参数，有个 默认值 None
    def forward(ctx, input, weight, bias=None):
        # input，weight 都已经变成了 Tensor
        # 用 ctx 把该存的存起来，留着 backward 的时候用
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())#weight转置
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # 由于 forward 只有一个 返回值，所以 backward 只需要一个参数 接收 梯度。
    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是 Variable 类型。
        # 在开头的地方将保存的 tensor 给 unpack 了
        # 然后 给 所有应该返回的 梯度 以 None 初始化。
        # saved_variables 返回的是 Variable！！！ 不是 Tensor 了。
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # needs_input_grad 检查是可选的。如果想使得 代码更简单的话，可以忽略。
        # 给不需要梯度的 参数返回梯度 不是一个错误。
        # 返回值 的个数 需要和 forward 形参的个数（不包含 ctx）一致
        # wx+b
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)#对x求导为m，乘上系数
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)#对w求导为x，乘上系数
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        # 梯度的顺序和 forward 形参的顺序要对应。
        return grad_input, grad_weight, grad_bias
```
关于 ctx:context上下文
- save_for_backward 只能存 tensor, None, 其余都不能存。
- save_for_backward 只保存 forward 的实参，或者 forward 的返回值。
```python
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None
```
上述演示了forward非Variable类型的实参。注意此时backward返回值数量也必须和forward参数数量一致(除了ctx)，非Variable或不需要计算梯度的，返回None代替。


```python
from torch.autograd import gradcheck
# Check gradients computed via small finite differences against analytical gradients

# 检查的是 inputs 中 requires_grad=True 的梯度，
# 一定要记得 double() 一下！！！！！！
input = (Variable(torch.randn(20, 20).double(), requires_grad=True),
             Variable(torch.randn(30, 20).double(), requires_grad=True),)
test = gradcheck(LinearFunction.apply, input, eps=1e-6, atol=1e-4)
# 如果通过，最后会打印一个 True
print(test)
```
上述检查定义的梯度计算是否正确。可以认为是一层神经网络，输入层为20个神经元，共20个样本；输出层30个神经元，每个神经元和输入层20个神经元相连接。计算得到的结果为20\*30规格。即每个样本得到一个30维的输出层向量。

# 理解
## Function的直观理解
- 在之前的介绍中，我们知道，Pytorch是利用Variable与Function来构建计算图的。回顾下Variable，Variable就像是计算图中的节点，保存计算结果（包括前向传播的激活值，反向传播的梯度），而Function就像计算图中的边，实现Variable的计算，并输出新的Variable。
- Function简单说就是对Variable的运算，如加减乘除，relu，pool等。但它不仅仅是简单的运算。与普通Python或者numpy的运算不同，Function是针对计算图，需要计算反向传播的梯度。因此他不仅需要进行该运算（forward过程），还需要保留前向传播的输入，中间计算的未激活值、激活值，并支持反向传播计算梯度。如果有写过神经网络的话，会发现进行反向传播计算前，都需要先进行一遍前向传播，记录未激活值和激活值，然后在计算梯度的时候用到。这两者是类似的。
- 在之前Variable的学习中，我们知道进行一次运算后，输出的Variable对应的creator就是其运行的计算，如y = relu(x), y.creator，就是relu这个Function。
- 我们可以对Function进行拓展，使其满足我们自己的需要，而拓展就需要自定义Function的forward运算，以及对应的backward运算，同时在forward中需要通过保存输入值用于backward。
- 总结，Function与Variable构成了pytorch的自动求导机制，它定义了各个Variable之间的计算关系。


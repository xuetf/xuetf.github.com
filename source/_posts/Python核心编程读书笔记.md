---
title: Python核心编程读书笔记
date: 2017-09-07 20:05:12
tags: [Python]
categories: 学习笔记
---
　　本文是Python核心编程的读书笔记。
<!--more-->
# Chapter 1 Welcome to Python
## 什么是Python
　　**Python is an elegant and robust programming language that delivers both the power and general applicability of traditional compiled languages with the ease of use of simpler scripting and interpreted languages.**
## 起源
　　贵泽.范.罗萨姆(Guido van Rossum)于1989年底创造了Python.1991年，Python发布了第一个公开发行版。
## 特点
- 高级语言(High Level)
- 面向对象(Object Oriented)
- 可升级(Scalable)：提供基本开发模块，可在此之上开发你的软件，当需要扩展和增长时，Python可插入性和模块化机构使得项目易于管理。
- 可扩展(Extensible)：可用其他语言编写Python扩展。如CPython和Jython。
- 可移植性(Portable)
- 易学(Easy to Learn)
- 易读(Easy to Read)
- 易维护(Easy to Maintain)
- 健壮性(Robust): 错误处理机制
- 高效快速的原型开发工具(Effective as a Rapid Prototyping Tool)：拥有面向其它系统接口，标准库和第三方模块完备。
- 内存管理器(A Memory Manager)
- 解释性和(字节)编译性:解释型语言，但是Python实际上是字节编译的(生成带有.pyc或.pyo的扩展名文件)，可以生成一种近似机器语言的中间形式。兼顾了编译型语言的性能和解释型语言的优点。

## Python和其他语言比较
### Python VS Perl
　　同：都是脚本语言.
　　异：Perl最大的优势在于它的字符串模式匹配能力，提供了强大的正则表达式匹配引擎，使得Perl实际上成为了一种用于过滤、识别和抽取字符串文本的语言。一直是开发web服务器端通用网关接口CGI网络程序最流行的语言，也是目前Linux中大多数脚本使用的语言。然而Perl语言晦涩、对符号语法过度使用，解读很困难。Pyhon适合开发大程序，吸收了Perl语言特点，也拥有强大的正则化引擎，同时易学易用。
### Python VS Java
　　同：面向对象。Jython开发联结了二者。Jython是用Java开发的Python解释器，即可以将Python代码解释成Java虚拟机懂的字节码，这意味着可以在只有Java虚拟机的环境中运行Python程序。在Jython脚本环境中，还可以处理Java对象，可以访问Java的标准库。
　　异：Java繁琐，Python简洁。
### Python VS Ruby
　　同：脚本语言。
　　异：Python是多编程范式混合，Ruby完全面向对象。Python有一个字节码解释器，Ruby没有。Python更加易读，Ruby可看作是面向对象的Perl。相对于Ruby on Rails,Python有自己的Web应用框架，比如Django和Turbogears.

　　可参考，[Python与其他语言比较][1]

## 其它实现
　　标准版的Python是用C来编译的，又叫CPython。除此之外，还有一些其它的Python实现。
### Java
　　前面也有提到过。使用Java语言写成的Python解释器Jython。Jython优势：
- 只要有Java虚拟机，就能运行Jython。
- 拥有访问Java包宇类库的能力。
- 为Java开发环境提供了脚本引擎。
- 能够很容易的测试Java类库。
- 提供访问Java原生异常处理的能力。
- 继承了JavaBeans特性和内省能力。
- 鼓励Python到Java的开发。
- GUI开发人员可以访问Java的AWT/SWing库
- 利用了Java原生垃圾收集器。
### .NET
　　IronPython的Python实现，可以在一个.NET应用程序中整合IronPython解释器来访问.NET对象。
### Stackless
　　CPython的一个局限就是每个Python函数调用都会产生一个C函数调用。这意味着同时产生的函数调用时有限制的，因此Python难以实现用户级的线程库和复杂递归应用，一旦超越这个限制，程序就会崩溃。可以通过使用一个stackless的Python实现来突破这个限制，使得一个C栈帧可以拥有任意数量的Python栈帧，这样就能够拥有无穷的函数调用，并能支持巨大数量的线程。　

# Chapter 2 Getting Started
## print输出
　　通常当你想看变量内容时，会在代码中使用print语句输出。不过在交互式解释器中，可以使用print语句显示变量的字符串表示，或者仅使用变量名查看该变量的原始值。
![python note][2]
　　注意，在仅用变量名时，输出的字符串被用单引号括起来了。这是为了让非字符串对象也能以字符串的方式显示在屏幕上-即它显示的是该对象的字符串表示，而不仅仅是字符串本身。究其本质，print语句调用str()函数显示对象，而交互式解释器调用repr()函数显示对象。
　　下划线(_)在解释器中有特别的含义，表示最后一个表达式的值。
　　Python的print语句，与字符串格式运算符(%)结合使用，可实现字符串替换功能。
```python
>>>print "%s is number %d!" % ("Python", 1)
Python is number 1!
```
　　%s替换字符串，%d替换整数，%f替换浮点数。
　　Print语句支持将输出重定向到文件。使用符号>>重定向输出。如下是将输出重定向到标准错误输出：
```python
import sys
print >> sys.stderr, 'Fatal error: invalid input!'
```
　　下面将输出重定向到日志文件的例子：
```python
logfile = open('/tmp/mulog.txt','a')
print >> logfile, 'Fatal error:invalid input!'
logfile.close()
```

## raw_input输入
　　raw_input()读取标准输入，并将读到的数据赋值给指定的变量，但是该变量是字符串，可以使用int()等函数进行类型转换。
```python
>>>user = raw)input('Enter login name:')
Enter login name: root
>>> Print 'Your login is:',user
Your login is: root
```
## help帮助
　　在学习Python过程中，如果需要得到一个生疏函数的帮助，只需要对它调用help(函数名)就可以得到帮助信息。

## 注释
　　使用\# 注释单行。
　　有一种叫做文档字符串的特别注释。可以在模块、类或者函数的起始添加一个字符串，起到在线文档的功能。

```python
def foo():
  "This is a doc string"
  return True
```
 　　
　　与普通注释不同，文档字符串可以在运行时访问，也可以用来自动生成文档。

## 数字
　　Python五种基本数字类型：
- int(有符号整数)
- long(长整数)：超过C语言中的long，类似Java的BigInteger
- bool(布尔值)
- float(浮点值)
- complex(复数)

## 字符串
　　支持成对单引号或双引号；也支持三引号(三个连续的单引号或双引号)，可以用来包含特殊字符。
　　使用索引运算符([])和切片运算符([:])可以得到子字符串。第一个字符索引为0，最后一个字符索引为-1。
　　\*号可以用于字符串重复。例如：
```python
>>> "python" * 2
'pythonpython'
```

## 列表和元组
　　可以将列表和元组当作普通的数组，可以保存任意数量任意类型的Python对象，唯一不同的是列表和元组可以同时存储不同类型的对象。
　　列表和元组几处重要的区别：列表元素用中括号([])包裹，元素的个数及元素的值可以改变。元组元素用小括号(())包裹，不可以更改(尽管内容可以更改)。元组可以看成是只读的列表，通过切片运算([:])可以得到子集。

## 字典
　　字典用于映射数据，键值对构成。几乎所有类型的Python对象都可以用作键，不过一般以字符串或数字字符串最为常用，值可以用任意Python对象，字典元素用大括号({})包裹。
```python
>>>aDict={'host':'earth'} # create dict
>>>aDict['port']=80 # add to dict 
>>>aDict
{'host':'earth','port':80}
>>>aDict.keys()
['host','[port']
>>>for key in aDict:
       print key,aDict[key]
host earth
port 80
```

## for循环和range()、enumerate()
　　for循环类似foreach功能，不能循环索引。要循环索引可以使用range(len(list))。
如果既要循环索引，又要循环元素，恶意使用enumerate()。
```python
>>>foo = ['a','b','c']
>>>for i,ch in enumerate(foo):
    print ch, '(%d)'  % i
a (0)
b (1)
c (2)
```
## 文件和内建函数open()
```python
handle = open(file_name, access_mode='r') #access mode:'r','w','a','b'
for eachLine in handle:
    print eachLine
handle.close()
```
## 错误和异常
```python
try:
	handle = open(file_name, access_mode='r') #access mode:'r','w','a','b'
	for eachLine in handle:
	    print eachLine, handle.close() # error
except IOError, e:
	print 'file open error:',e

```
　　也可以通过raise语句故意引发一个异常。

## 类
```python
class ClassName(base_class[es]):
    "optional documentation string"
    static_member_declarations
    method_declarations
```
　　使用class关键字定义类，可以提供一个可选的父类，如果没有合适的父类，那就使用object作为父类。
```python
class FooClass(object):
    """my very first class: FooClass"""
    version = 0.1 # class (data) attribute
    def __init__(self, nm='John Doe'):
        """constructor"""
        self.name = nm # class instance (data) attribute
        print 'Created a class instance for', nm
    def showname(self):
        """display instance attribute and class name"""
        print 'Your name is', self.name
        print 'My name is', self.__class__.__name__
    def showver(self):
        """display class(static) attribute"""
        print self.version # references FooClass.version
    def addMe2Me(self, x): # does not use 'self'
        """apply + operation to argument"""
        return x + x
# 创建类实例
>>>foo1 = FooClass()
Created a class instance for John Doe
```
　　\_\_init\_\_可以当成构造函数，不过不像其他语言的构造函数，它并不会创建实例，它仅仅是你的对象创建后执行的第一个方法，做一些必要的初始化工作。\_\_init\_\_有个默认参数self。
　　self.\_\_class\_\_.name显示类名。

## 模块
　　一个Python源文件就是一个模块，模块不带.py后缀。可以使用import语句导入都其他模块中使用。

## 实用的函数
- dir([obj]):显示对象的属性，如果没有提供参数，则显示全局变量的名字。
- help([obj]):以一种整齐美观的形式显示对象的文档字符串，如果没有提供任何参
数，则会进入交互式帮助。
- int(obj): 对象转整数
- len(obj): 对象长度
- open(fn,mode):打开文件
- range([start,]stop[,step]):返回一个整数列表，起始值为start，结束值为stop-1,步阶是step。start默认为0，step默认为1.
- raw_input(str):等待用户输入一个字符串，str用于提示信息。
- str(obj): 对象转字符串
- type(obj): 返回对象的类型，返回值是一个type对象。

# Chapter 3 Python基础
## 语句和语法
　　\#是注释，\是换行，;可用于同一行书写多个表达式
## 赋值
　　多元赋值:
```python
>>>x, y, z= 1, 2, 'a'#实际上是元组赋值，即(x, y, z)=(1, 2, 'a')
```
## 下划线标识符
- \_xxx:可以看作是私有的，不能通过"from module import *"导入。
- \_\_xxx\_\_:系统定义名字
- \_\_xxx：类中的私有变量名

## 基本风格指南
　　python提供了一个机制，可以通过\_\_doc\_\_特别变量，动态获得文档字串。在模块，类声明，或函数声明中**第一个没有赋值的字符串**可以通过属性obj.\_\_doc\_\_来进行访问，其中obj是一个模块，类或函数的名字。
　　使用4个空格缩进。

## 内存管理
　　引用计数。记录所有使用中的对象各有多个引用。引用计数为0时，它被垃圾回收。
### 引用计数增加
　　当对象被创建并将其引用赋值给变量时，该对象的引用计数设置为1.
　　当同一个对象的引用又被赋值给其他变量时，或作为参数传递给函数，方法或类实例时，或者被赋值为一个窗口对象(list,tuple等)的成员时，该对象新的引用被创建，则该对象的引用计数加1。

### 引用计数减少
　　当对象的引用被销毁时，引用计数会减小。最明显的例子就是当引用离开其作用范围时，这种情况最经常出现在函数运行结束时，所有局部变量都自动销毁，对象的引用计数也随之减少。
　　当变量被赋值给另外一个对象时，原对象的引用计数也会自动减少1。
```python
foo = 'xyz'
bar = foo
foo = 123
```
　　如上，当字符串对象“xyz”被创建并赋值给foo时，它的引用计数为，当增加一个别名bar时，引用计数变为2.不过当foo被重新赋值给整数对象123时，‘xyz’对象的引用计数自动减少1，又重新变成了1，此时只有bar是指向“xyz”。
　　其他造成对象引用计算减少的方式包括del语句删除一个变量。或者当一个对象被移出一个窗口对象时，或者该窗口对象本身的引用计数变成了0时。　　
　　总结一下一个对象的引用计数在以下情况会减少：
- 一个本地引用离开了作用范围。比如foobar()函数结束时。
- 对象的别名被显示销毁。del y
- 对象的一个别名被赋值给其它的对象。 x=123
- 对象被从一个窗口对象中移除：myList.remove(x)
- 窗口对象本身被销毁：del myList
　　del看一个例子：
```python
x=3.14 #创建3.14对象并赋值给x
y=x
```
　　执行del y会产生两个结果：1）从现在的名字空间中删除y。2）x的引用计数减少1。
　　引申一步，紧接着执行del x会删除该数值对象的最后一个引用，也就是该对象的引用计数会减少为0，这会导致该对象从此无法访问或无法抵达。从此刻起，该对象就成为垃圾回收机制的回收对象。注意**任何追踪或调试程序会给一个对象增加一个额外的引用**，这会推迟该对象被回收的时间。
### 垃圾回收
　　垃圾回收器会回收引用计数为0的对象，也负责检查那些虽然引用计数大于0但也应该被销毁的对象，特定情形会导致循环引用。
　　一个循环引用发生在当你有至少两个对象相互引用时，也就是所有的引用都消失了，这些引用仍然存在，这说明只靠计数是不够的。Python垃圾回收器实际上是一个引用计数器和一个循环垃圾回收器。Python会留心被分配的总量很大（即未通过引用计数销毁的那些）的对象，这种情况下，解释器会暂停下来，试图清理所有未引用的循环。

# Chapter 4 Python对象
　　未完待续......


















[1]: https://wiki.python.org/moin/LanguageComparisons
[2]: /picture/machine-learning/python-note1.png
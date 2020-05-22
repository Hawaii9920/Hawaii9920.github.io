 

**深度学习算法Pytorch****基础**

 

**教案**

 

 

 

| **课程教案版本** | **日期**     | **备注** |
| ---------------- | ------------ | -------- |
| **V1.0**         | **20190515** |          |
|                  |              |          |

 



# **Pytorch基础教程1**

Url: <https://pytorch.org/>

#1.安装

Conda安装

conda install pytorch-cpu torchvision-cpu -c pytorch

Pip安装

pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp36-cp36m-win_amd64.whl

 

目标：PyTorch是使用GPU和CPU优化的深度学习张量库。

l 在高层次上理解PyTorch的Tensor库和神经网络。

l 训练一个小型神经网络对图像进行分类

l Make sure you have the [**torch**](https://github.com/pytorch/pytorch) and [**torchvision**](https://github.com/pytorch/vision) packages installed.

# **2.什么是Pytorch？**

这是一个基于Python的科学计算软件包，针对两组受众：

* NumPy的替代品，可以使用GPU的强大功能

* 深入学习研究平台，提供最大的灵活性和速度

张量:张量与NumPy的ndarray类似，另外还有Tensors也可用于GPU以加速计算。

```python
from future import print_function import torch
```

构造一个未初始化的5x3矩阵：

```python
x = torch.empty(5, 3) 
print(x)
```



输出：

```
tensor([[0., 0., 0.],         
[0. , 0., 0.],         
[0., 0., 0.],         
[0., 0., 0.],         
[0., 0., 0.]])

```



构造一个随机初始化的矩阵：

```
x = torch.rand(5, 3) 
print(x) 
```

输出：

```
tensor([[0.5728, 0.5375, 0.0494],         
[0.2820, 0.1853, 0.8619],         
[0.0856, 0.8380, 0.8117],         
[0.7959, 0.8802, 0.3610],         
[0.4440, 0.4028, 0.2289]])
```



构造一个矩阵填充的零和dtype long：

```
x = torch.zeros(5, 3, dtype=torch.long) 
print(x)
```



输出：

```
tensor([[0, 0, 0],         
[0, 0, 0],         
[0, 0, 0],         
[0, 0, 0],         
[0, 0, 0]])
```



直接从数据构造张量：

```python
x = torch.tensor([5.5, 3]) 
print(x)
```



日期：

```
tensor([5.5000, 3.0000])
```



或者根据现有的张量创建张量。除非用户提供新值，否则这些方法将重用输入张量的属性，例如dtype

```
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes print(x) x = torch.randn_like(x, dtype=torch.float)    # override dtype! 
print(x)                       # result has the same size
```



日期：

```
tensor([[1., 1., 1.],         
[1., 1., 1.],         
[1., 1., 1.],         
[1., 1., 1.],         
[1., 1., 1.]], dtype=torch.float64) 
tensor([[ 0.3928, -0.4377, -0.6426],         
[ 0.6000,  0.1942, -0.9790],         
[-3.0629,  0.2410, -1.5378],         
[ 0.0219,  0.5899,  0.8386],         
[-0.1540,  0.2724,  0.3881]])
```

得到它的大小：

```
print(x.size()) 
```

日期：

```python
torch.Size([5, 3])
```

注意：torch.Size 实际上是一个元组，因此它支持所有元组操作。

 

# **2.Pytorch**操作

操作有多种语法。在下面的示例中，我们将查看添加操作。

增加：语法1

```
y = torch.rand(5, 3) 
print(x + y) 
```

操作：

```
tensor([[ 1.0653,  0.1367, -0.1789],         
[ 1.2751,  0.3061, -0.2860],         
[-2.9511,  0.3313, -1.5280],         
[ 0.5614,  1.4668,  1.7641],         
[ 0.1087,  0.8200,  0.5692]])
```



增加：语法2

```
print(torch.add(x, y)) 
```



日期：

```
tensor([[ 1.0653,  0.1367, -0.1789],         
[ 1.2751,  0.3061, -0.2860],         
[-2.9511,  0.3313, -1.5280],         
[ 0.5614,  1.4668,  1.7641],         
[ 0.1087,  0.8200,  0.5692]])
```



增加：提供输出张量作为参数

```
result = torch.empty(5, 3) 
torch.add(x, y, out=result) print(result) 
```



日期：

```
tensor([[ 1.0653,  0.1367, -0.1789],         
[ 1.2751,  0.3061, -0.2860],         
[-2.9511,  0.3313, -1.5280],         
[ 0.5614,  1.4668,  1.7641],         
[ 0.1087,  0.8200,  0.5692]])
```



 

增加：就地

```
# adds x to y y.add_(x) print(y) 
```



输出：

```
tensor([[ 1.0653,  0.1367, -0.1789],         

[ 1.2751,  0.3061, -0.2860],         

[-2.9511,  0.3313, -1.5280],         

[ 0.5614,  1.4668,  1.7641],         

[ 0.1087,  0.8200,  0.5692]])

```



任何使原位张量变形的操作都是用_。后固定的。例如：x.copy_(y)，x.t_()，将改变x。

你可以使用标准的NumPy索引

print(x[:, 1]) 

日期：

tensor([-0.4377,  0.1942,  0.2410,  0.5899,  0.2724]) 

调整大小：如果要调整张量/重塑张量，可以使用torch.view：

```
x = torch.randn(4, 4) 
y = x.view(16) 
z = x.view(-1, 8)  
# the size -1 is inferred from other dimensions 
print(x.size(), y.size(), z.size()) 
```



输出：

```python
torch.Size([4, 4]) 
torch.Size([16]) 
torch.Size([2, 8])
```



如果你有一个元素张量，用于.item()获取值作为Python数字

```
x = torch.randn(1) 
print(x) 
print(x.item()) 
```



日期：

```
tensor([-1.7816]) 
-1.7815848588943481
```



 

 

# **3.**NumPy Bridge

将Torch Tensor转换为NumPy阵列（反之亦然）是一件轻而易举的事。

Torch Tensor和NumPy阵列将共享其底层内存位置，更改一个将改变另一个。

将Torch Tensor转换为NumPy数组

```
a = torch.ones(5) 
print(a) 
```

输出：

```
tensor([1., 1., 1., 1., 1.])
```

转化为numpy：

```python
b = a.numpy() 
print(b) 
```

输出：

[1. 1. 1. 1. 1.] 

了解numpy数组的值如何变化。

```
a.add_(1) 
print(a) 
print(b) 
```

输出：

tensor([2., 2., 2., 2., 2.]) 

[2. 2. 2. 2. 2.]

将NumPy数组转换为Torch Tensor

* 了解更改np阵列如何自动更改Torch Tensor

```
import numpy as np 
a = np.ones(5) 
b = torch.from_numpy(a) 
np.add(a, 1, out=a) 
print(a) 
print(b)
```

输出：

[2. 2. 2. 2. 2.] 

tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

除了CharTensor之外，CPU上的所有Tensors都支持转换为NumPy并返回。

#**4.Autograd：自动求导**

张量

torch.Tensor是包的核心类。**如果将其属性设置** **.requires_grad**为True，则会开始跟踪其上的所有操作。完成计算后，您可以调用.backward()并自动计算所有渐变。该张量的梯度将累积到.grad属性中。

要阻止张量跟踪历史记录，您可以调用.detach()它将其从计算历史记录中分离出来，并防止将来的计算被跟踪。

要防止跟踪历史记录（和使用内存），您还可以将代码块包装在其中。这在评估模型时尤其有用，因为模型可能具有可训练的参数 ，但我们不需要梯度。

```
with torch.no_grad():
		requires_grad=True
```



还有一个类对于autograd实现非常重要 -Function。

Tensor并且Function互相连接并构建一个非循环图，它编码完整的计算历史。**每个张量都有一个.grad_fn**属性，该属性引用Function已创建的属性Tensor（除了用户创建的张量 ）。

如果你想计算导数，你可以调用.backward()的Tensor。如果Tensor是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 匹配形状的张量的参数。

```
import torch 
```

创建一个张量并设置requires_grad=True为跟踪计算

```python
x = torch.ones(2, 2, requires_grad=True) 
print(x) 
```

输出：

```
tensor([[1., 1.],         [1., 1.]], requires_grad=True)
```

做一个张量操作：

```
y = x + 2 
print(y) 
```

输出：

```
tensor([[3., 3.],         [3., 3.]], grad_fn=<AddBackward0>) 
```

y是作为一个操作的结果创建的，所以它有一个grad_fn。

```
print(y.grad_fn) 
```

输出：

<AddBackward0 object at 0x7fa2bfab3ac8> 

做更多的操作 y

z **=** y ***** y ***** 3 

out **=** z**.**mean() 

print(z, out) 

输出：

tensor([[27., 27.],         [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>) 

.requires_grad_( ... )requires_grad 就地改变现有的Tensor 旗帜。False如果没有给出，输入标志默认为。

a **=** torch**.**randn(2, 2)

 a **=** ((a ***** 3) **/** (a **-** 1)) 

print(a**.**requires_grad) a**.**requires_grad_(**True**)

print(a**.**requires_grad) b **=** (a ***** a)**.**sum() print(b**.**grad_fn) 

输出：

False True <SumBackward0 object at 0x7fa2bfac57b8>

渐变

我们现在回来了。因为out包含单个标量，out.backward()相当于out.backward(torch.tensor(1.))。

out**.**backward() 

打印渐变d（out）/ dx

print(x**.**grad) 

日期：

tensor([[4.5000, 4.5000],         [4.5000, 4.5000]])

![1564551690874](assets/1564551690874.png) 

```
x = torch.randn(3, requires_grad=True) 
y = x * 2 
while y.data.norm() < 1000:     
	y = y * 2 
	print(y) 

```

输出：

tensor([-333.7958, -287.0981, 1274.5677], grad_fn=<MulBackward0>) 

现在在这种情况下y不再是标量。torch.autograd 无法直接计算完整的雅可比行列式，但如果我们只想要矢量雅可比产品，只需将向量传递给 backward参数：

```
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float) 
y.backward(v) 
print(x.grad) 
```

输出：

tensor([2.0480e+02, 2.0480e+03, 2.0480e-01])

您还可以.requires_grad=True通过包装代码块来 停止在Tensors上跟踪历史记录的autograd通过with torch.no_grad():

```scala
print(x.requires_grad) 
print((x ** 2).requires_grad) 
with torch.no_grad():     
	print((x ** 2).requires_grad) 

```

输出：

```
True 
True 
False 
```



 

#**5.神经网络**

可以使用torch.nn包构造神经网络。

现在你已经看到了autograd，nn取决于 autograd定义模型并区分它们。一个nn.Module包含层，和一种方法forward(input)，它返回output。

例如，查看对数字图像进行分类的网络：

![1564551701075](assets/1564551701075.png) 

它是一个简单的前馈网络。它接受输入，一个接一个地通过几个层输入，然后最终给出输出。

神经网络的典型训练程序如下：

· 定义具有一些可学习参数（或权重）的神经网络

· 迭代输入数据集

· 通过网络处理输入

· 计算损失（输出距离正确多远）

· 将渐变传播回网络参数

· 通常使用简单的更新规则更新网络权重： weight = weight - learning_rate * gradient

 

定义网络

我们来定义这个网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```



输出：

```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```



您只需定义forward函数，backward 自动被实现（计算渐变的位置）autograd。您可以在forward函数中使用任何Tensor操作。

模型的可学习参数由返回 net.parameters()

```
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

输出：

10 

torch.Size([6, 1, 5, 5])



让我们尝试一个随机的32x32输入。注意：此网络（LeNet）的预期输入大小为32x32。要在MNIST数据集上使用此网络，请将数据集中的图像调整为32x32。

input **=** torch**.**randn(1, 1, 32, 32) out **=** net(input) print(out) 

日期：

tensor([[-0.0027,  0.0397, -0.0506,  0.1499, -0.0068, -0.0074,  0.0797, -0.0323,          -0.0592,  0.0341]], grad_fn=<AddmmBackward>) 

使用随机梯度将所有参数和反向的梯度缓冲区归零：

```
net.zero_grad() 
out.backward(torch.randn(1, 10)) 
```

注意

torch.nn仅支持mini批次。整个torch.nn 软件包仅支持小批量样本的输入，而不是单个样本。

例如，nn.Conv2d将采用4D Tensor of 。nSamples x nChannels x Height x Width

如果您有一个样本，只需使用input.unsqueeze(0)添加假批量维度。

 

 

在继续之前，让我们回顾一下你到目前为止看到的所有课程。

**概括：**

· torch.Tensor- 支持自动编程操作*的多维数组*backward()。也*保持*张量与张量。

· nn.Module - 神经网络模块。*方便的封装参数的方法*，使用帮助程序将它们移动到GPU，导出，加载等。

· nn.Parameter- 一种Tensor，*在被指定为a的属性时自动注册为参数* Module。

· autograd.Function- 实现*自动编程操作的前向和后向定义*。每个Tensor操作至少创建一个Function节点，该节点连接到创建Tensor和*编码其历史记录的*函数。

**在这一节上，我们涵盖了：**

* 定义神经网络

* 处理输入并向后调用

**还是左：**

* 计算损失

* 更新网络权重



##5.1损失函数

损失函数采用（输出，目标）输入对，并计算估计输出距目标的距离的值。

nn包下有几种不同的 [损失函数](#loss-functions)。一个简单的损失是：nn.MSELoss它计算输入和目标之间的均方误差。

例如：

```
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```



输出：

tensor(0.8012, grad_fn=<MseLossBackward>) 

现在，如果您按照loss向后方向，使用其 .grad_fn属性，您将看到如下所示的计算图：

```python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```



因此，当我们调用时loss.backward()，整个图形会随着损失而自动微分，并且图形中的所有张量都requires_grad=True将.grad使用渐变累积其Tensor。

为了说明，让我们向后退几步：

```python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```



输出：

```
<MseLossBackward object at 0x7fbabaa439b0>
<AddmmBackward object at 0x7fbabaa43ac8>
<AccumulateGrad object at 0x7fbabaa43ac8>
```



##5.2Backprop

要反向传播错误，我们所要做的就是loss.backward()。您需要清除现有渐变，否则渐变将累积到现有渐变。

现在loss.backward()，看一下conv1在向后之前和之后的偏差梯度。

```
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

输出：

```
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0040,  0.0309, -0.0133,  0.0081,  0.0409, -0.0103])
```

现在，我们已经看到了如何使用损失函数。

**稍后阅读：**

神经网络包包含形成深度神经网络构建块的各种模块和损失函数。带有文档的完整列表[在这里](https://pytorch.org/docs/nn)。

 

**唯一要学习的是：**

* 更新网络权重

 

更新权重

**手动实现**

* 实践中使用的最简单的更新规则是随机梯度下降（SGD）：
  * weight = weight - learning_rate * gradient

我们可以使用简单的python代码实现它：

```scala
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```



**自动实现** 

但是，当您使用神经网络时，您希望使用各种不同的更新规则，例如SGD，Nesterov-SGD，Adam，RMSProp等。为了实现这一点，我们构建了一个小包：torch.optim它实现了所有这些方法。使用它非常简单：

```
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



注意：观察如何使用手动将梯度缓冲区设置为零 optimizer.zero_grad()。这是因为渐变是按[Backprop](#backprop)部分中的说明累积的。

 

#6.训练分类器(完整例子)

到此。您已经了解了如何定义神经网络，计算损耗并更新网络权重。

现在你可能在想，数据怎么样？

通常，当您必须处理图像，文本，音频或视频数据时，您可以使用标准的python包将数据加载到numpy数组中。然后你可以将这个数组转换成一个torch.*Tensor。

* 对于图像，Pillow，OpenCV等软件包很有用

* 对于音频，包括scipy和librosa

* 对于文本，无论是原始Python还是基于Cython的加载，还是NLTK和SpaCy都很有用

特别是对于视觉，我们创建了一个名为的包 torchvision，它包含用于常见数据集的数据加载器，如Imagenet，CIFAR10，MNIST等，以及用于图像的数据转换器，即 torchvision.datasets和torch.utils.data.DataLoader。

这提供了极大的便利并避免编写样板代码。

​		在本教程中，我们将使用CIFAR10数据集。它有类：'飞机'，'汽车'，'鸟'，'猫'，'鹿'，'狗'，'青蛙'，'马'，'船'，'卡车'。CIFAR-10中的图像尺寸为**3x32x32，即尺寸为32x32像素的3通道彩色图像。**

 

![img](file:///C:\Users\zhao-chj\AppData\Local\Temp\ksohtml17188\wps3.jpg) 

cifar10

训练图像分类器

我们将按顺序执行以下步骤：

* 使用加载和标准化CIFAR10训练和测试数据集 torchvision

* 定义卷积神经网络

* 定义损失函数

* 在训练数据上训练网络

* 在测试数据上测试网络

 

##6.1.加载和标准化CIFAR10

使用torchvision，加载CIFAR10非常容易。

```
import torch
import torchvision
import torchvision.transforms as transforms
```

torchvision数据集的输出是范围[0,1]的PILImage图像。我们将它们转换为归一化范围的张量[-1,1]。

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```



输出：

Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz Files already downloaded and verified

让我们展示一些训练图像，以获得乐趣。

```
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```



![img](file:///C:\Users\zhao-chj\AppData\Local\Temp\ksohtml17188\wps4.jpg) 

 

输出：

car  ship  ship  frog

 

##6.2.定义卷积神经网络

从神经网络部分复制神经网络并修改它以获取3通道图像（而不是定义的1通道图像）。

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```



##6.3.定义Loss函数和优化器

让我们使用分类交叉熵损失和SGD动量。

```
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



 

4.训练网络

事情开始变得有趣了。我们只需循环遍历数据迭代器，并将输入提供给网络并进行优化。

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```



 

日期：

```python
[1,  2000] loss: 2.220
[1,  4000] loss: 1.850
[1,  6000] loss: 1.646
[1,  8000] loss: 1.560
[1, 10000] loss: 1.515
[1, 12000] loss: 1.455
[2,  2000] loss: 1.403
[2,  4000] loss: 1.374
[2,  6000] loss: 1.375
[2,  8000] loss: 1.328
[2, 10000] loss: 1.310
[2, 12000] loss: 1.298
Finished Training
```



 

##6.5.在测试数据上测试网络

我们已经在训练数据集上训练了2个epoch。但我们需要检查网络是否学有效果

我们将通过预测神经网络输出的类标签来检查这一点，并根据地面实况进行检查。如果预测正确，我们将样本添加到正确预测列表中。

好的，第一步。让我们从测试集中显示一个图像以熟悉。

```
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```



![img](file:///C:\Users\zhao-chj\AppData\Local\Temp\ksohtml17188\wps5.jpg) 

日期：

GroundTruth:    cat  ship  ship plane 

好的，现在让我们看看神经网络认为上面这些例子是什么：

outputs **=** net(images) 

输出是10类的能量。一个类的能量越高，网络认为图像是特定类的越多。那么，让我们得到最高能量的指数：

```
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
```



输出：

Predicted:    cat  ship truck plane 

结果似乎很好。

让我们看看网络如何在整个数据集上执行。

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```



输出：

Accuracy of the network on the 10000 test images: 54 %

看起来好于随机猜测结果，这是10％的准确性（从10个类中随机挑选一个类）。证明网络学到了一些东西。

 

 

嗯，什么是表现良好的类，以及表现不佳的类：

```
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```



输出：

```
Accuracy of plane : 66 %
Accuracy of   car : 62 %
Accuracy of  bird : 42 %
Accuracy of   cat : 17 %
Accuracy of  deer : 47 %
Accuracy of   dog : 63 %
Accuracy of  frog : 67 %
Accuracy of horse : 57 %
Accuracy of  ship : 63 %
Accuracy of truck : 68 %
```

好的，接下来呢？

我们如何在GPU上运行这些神经网络？

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
```

输出：

```
cuda:0
```

将变量转到cuda tensor上

```
net.to(device)
```

Remember that you will have to send the inputs and targets at every step to the GPU too:

```
inputs, labels = data[0].to(device), data[1].to(device)
```



在多个GPU上进行培训

如果您想使用所有GPU查看更多MASSIVE加速，请查看[可选：数据并行](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)。

我下一步去哪儿？

· [训练神经网络玩视频游戏](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

· [在imagenet上培训最先进的ResNet网络](https://github.com/pytorch/examples/tree/master/imagenet)

· [使用Generative Adversarial Networks训练面部生成器](https://github.com/pytorch/examples/tree/master/dcgan)

· [使用Recurrent LSTM网络训练单词级语言模型](https://github.com/pytorch/examples/tree/master/word_language_model)

· [更多例子](https://github.com/pytorch/examples)



# 7.数据并行

我们将学习如何使用多个GPU `DataParallel`。

使用PyTorch非常容易使用GPU。您可以将模型放在GPU上：

```
device = torch.device("cuda:0")
model.to(device)
```

然后，您可以将所有张量复制到GPU：

```
mytensor = my_tensor.to(device)
```

请注意，只是调用`my_tensor.to(device)`返回`my_tensor`GPU上的新副本 而不是重写`my_tensor`。您需要将其分配给新的张量并在GPU上使用该张量。

在多个GPU上执行前向，后向传播是很自然的。但是，Pytorch默认只使用一个GPU。通过使用`DataParallel`以下方式并行运行模型，您可以轻松地在多个GPU上运行操作 ：

```
model = nn.DataParallel(model)
```

这是本教程背后的核心。我们将在下面更详细地探讨它。

* 导入和参数

导入PyTorch模块并定义参数。

```
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100
```

设备

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 7.1虚拟数据集

制作一个虚拟（随机）数据集。你只需要实现getitem

```
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
```

## 7.2简单模型

对于演示，我们的模型只获取输入，执行线性操作并提供输出。但是，您可以`DataParallel`在任何型号（CNN，RNN，胶囊网等）上使用

我们在模型中放置了一个print语句来监视输入和输出张量的大小。请注意批次0级打印的内容。

```
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

## 7.3创建模型和DataParallel

这是本教程的核心部分。首先，我们需要创建一个模型实例并检查我们是否有多个GPU。如果我们有多个GPU，我们可以使用包装我们的模型`nn.DataParallel`。然后我们可以将我们的模型放在GPU上 `model.to(device)`

```
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```

日期：

```
Let's use 2 GPUs!
```

## 7.4运行模型

现在我们可以看到输入和输出张量的大小。

```
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
```

日期：

```
In Model: input size    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
 torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

## 7.5结果

如果您没有GPU或一个GPU，当我们批量输入30个输出和30个输出时，模型将获得30并按预期输出30。但是如果你有多个GPU，那么你可以得到这样的结果。

### 7.5.1 2个GPU

如果你有2，你会看到：

```
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 7.5.2 3个GPU

如果你有3个GPU，你会看到：

 

```
Let's use 3 GPUs!
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

### 7.5.3 8个GPU

如果你有8，你会看到：

```
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

## 7.6总结

DataParallel自动拆分数据并将作业订单发送到多个GPU上的多个模型。在每个模型完成其工作后，DataParallel会在将结果返回给您之前收集并合并结果。

有关详细信息，请查看 <https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html>。

 

 

 

 

 

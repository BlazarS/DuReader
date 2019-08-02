# paddl.fulid的一些知识  
## PaddlePaddle的数据读取方式  
处理完数据后如何把数据放到模型里去训练呢？我们知道，基本的方法一般有两种：

* 一次性加载到内存：模型训练时直接从内存中取数据，不需要大量的IO消耗，速度快，适合少量数据。  
* 加载到磁盘/HDFS/共享存储等：这样不用占用内存空间，在处理大量数据时一般采取这种方式，但是缺点是每次数据加载进来也是一次IO的开销，非常影响速度。  

在PaddlePaddle中我们可以有三种模式来读取数据：分别是**reader**、**reader creator**和**reader decorator**,这三者有什么区别呢？  
* reader：从本地、网络、分布式文件系统HDFS等读取数据，也可随机生成数据，并返回一个或多个数据项的函数。
* reader creator：一个返回reader的函数。
* reader decorator：一个函数，接收一个或多个reader,并返回一个reader。  
* batch reader是一个函数，它读取数据（从reader、文件、网络、随机数生成器中）并生成一批数据项。  
### Data Reader Interface  
> paddle.reader.shuffle(reader,buf_size)

创建数据读取器，该reader的数据输出将被无序排列。  
由原始reader创建的迭代器的输出将被缓冲到shuffle缓冲区，然后进行打乱。  
打乱缓冲区的大小由参数buf_size决定。  

参数：  
* reader(callable)--输出会被打乱的原始reader  
* buf_size(int)--打乱缓冲器的大小  
返回：
* 输出会被打乱的reader  
* 类型：callable  
### fluid.layer  
>paddle.fluid.layers.data(name,shape,
append,batch_size=True,dtype='float32',lod_level=0,type=VarType.LOD_TENSOR,stop_gradient=True)  

数据层(Data Layer)  
该功能接收输入数据，判断是否需要以minibatch方式返回数据，然后使用辅助函数创建全局变量。该全局变量可由计算图中的所有operator访问。  
这个函数的所有输入变量都作为本地变量传递给LayerHelper构造函数。  
参数：  
* name(str) 函数名或函数别名
* shape(list) 声明维度信息的list。如果没有`append_batch_size`为True且内部没有维度-1，则应将其视为每个样本的形状。否则，应将其是为batch数据的形状。  
* append_batch_size(bool)
>1.如果为真，则在维度shape的开头插入-1 “如果shape=[1],则输出shape为[-1,1].”  
2.如果维度shape包含-1，比如shape=[-1,1], “append_batch_size则为False（表示无效）”  

* dtype (basestring)-数据类型：float32,float_16,int等  
* type (VarType)-输出类型。默认为LOD_TENSOR  
* lod_level (int)-LoD层。0表示输入数据不是一个序列  
* stop_gradient (bool)-布尔类型，提示是否应该停止计算梯度  
>paddle.fluid.layers.fc(input, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, is_test=False, name=None)  

**全连接层**  
该函数在神经网络中建立一个全连接层。 它可以将一个或多个tensor（ input 可以是一个list或者Variable，详见参数说明）作为自己的输入，并为每个输入的tensor创立一个变量，称为“权”（weights），等价于一个从每个输入单元到每个输出单元的全连接权矩阵。FC层用每个tensor和它对应的权相乘得到形状为[M, size]输出tensor，M是批大小。如果有多个输入tensor，那么形状为[M, size]的多个输出张量的结果将会被加起来。如果 bias_attr 非空，则会新创建一个偏向变量（bias variable），并把它加入到输出结果的运算中。最后，如果 act 非空，它也会加入最终输出的计算中。  
当输入为单个张量：
                *Out=Act(XW+b)*   
当输入为多个张量：
            
上述等式中：  
        N ：输入的数目,如果输入是变量列表，N等于len（input）  
        Xi : 第i个输入的tensor  
        Wi ：对应第i个输入张量的第i个权重矩阵  
        b ：该层创立的bias参数  
        Act : activation function(激励函数)  
        Out : 输出tensor  
```Python
Given:
    data_1.data = [[[0.1, 0.2],
                   [0.3, 0.4]]]
    data_1.shape = (1, 2, 2) # 1 is batch_size

    data_2 = [[[0.1, 0.2, 0.3]]]
    data_2.shape = (1, 1, 3)

    out = fluid.layers.fc(input=[data_1, data_2], size=2)

Then:
    out.data = [[0.18669507, 0.1893476]]
    out.shape = (1, 2)
```  
实际上线性回归模式用的就是全连接层，例如房价预测项目。  
### 其他  
本小结记录一些从项目示例中看到的函数或者方法思想，日后归类。  
#### paddle.batch()
> paddle.batch()

这是一个我在房价预测模型中碰到的函数，batch的意思是“一批”。`paddle.batch()`的作用就是生成一个batch reader。  
`paddle.batch()`的源代码为：  
```python
__all__ = ['batch']


def batch(reader, batch_size, drop_last=False):
    """
    Create a batched reader.
    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
    :type drop_last: bool
    :return: the batched reader.
    :rtype: callable
    """

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if drop_last == False and len(b) != 0:
            yield b

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))

    return batch_reader
```  

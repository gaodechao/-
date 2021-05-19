##### Accelerating Graph Sampling for Graph Machine Learning using  GPUs 

###### Abstract

 图数据的几种表示学习算法，如DeepWalk、node2vec和GraphSAGE，对图进行采样，以产生适合训练DNN的mini-batches，然而，采样时间可能占训练时间的很大一部分，并且现有的系统不能有效地并行化采样。

采样似乎更适合GPU加速

<b>NextDoor</b>：设计用于在gpu上有效执行图形采样的系统。但图的不规则性使得GPU资源难以有效利用

NextDoor采用了一种新的图采样方法，并行传输(transit-parallelism)，它允许边的负载平衡和缓存。

NextDoor为终端用户提供了编写各种图形采样算法的高级抽象功能

<b>Introduction</b>

**该算法必须精心设计，以确保定期计算和内存访问，这是挑战性的不规则图**

它们通过查找每个样本顶点的邻近点来并行增长所有样本。 在这些系统中会导致两个问题

1.  不规则的内存访问和发散的控制流，因为连续的线程可以访问不同顶点的邻居，
2. 并行度较低，因为对样本中所有顶点的计算是由负责增长样本的线程串行执行的。

NextDoor，第一个在gpu上执行高效图采样的系统。

NextDoor通过根据与传输顶点关联的样本数量将它们分配到不同的内核，有效地平衡了跨传输顶点的负载。

每个内核都使用不同的调度和缓存策略来最大限度地使用执行资源和内存层次结构。

NextDoor有一个高级API，它抽象了在GPU上实现采样的低级细节，并使ML域专家能够用很少行代码编写高效的图采样算法。

transit-parallelism：

transit vertex：在transit-parallelism中的并行基本单元，它是一个顶点，其邻居可以被添加到图的一个或多个样本中。

**Contribution**

- 一个高级API，用于构建图形采样算法，并在gpu上高效执行
- 在gpu上执行图形采样的一种新的并行传输范式
- NextDoor，它利用了传输并行性，并添加了传输邻接表的负载平衡和缓存技术

**背景和Motivation**



**GPU性能要求**

1. 每个SM都有一个私有内存，称为共享内存，仅适用于分配给该SM的线程块。
2. GPU具有全局内存，所有SM均可访问。

SM从线程块调度一个线程子集，称为warp，warp通常由32个具有连续线程ID的线程组成

* 最小化*warp divergence*是在gpu上实现高性能的关键。
* 跨线程块平衡负载也很重要。
* 最后，GPU程序必须显式地选择使用共享内存或全局内存，并在可能时使用共享内存以最大化性能

GPU可以通过合并同一warp合并多个内存访问来提供对全局内存的高带宽访问。



**图采样的抽象**

该算法迭代地增长每个样本，以在一系列步骤中包含额外的顶点，其输出是扩展样本的最后一组。在每个步骤中，采样算法对每个样本执行以下操作：

1. 迭代的一次采样一个顶点，并将其添加到样本中。
2. 可以通过提供用户定义的函数来表示图采样应用程序。

samplingType函数描述了我们采样新顶点的粒度。抽样方式有两种类型：

1. *Individual Transit Sampling:*下一个函数是每次运输执行固定的次数。它可以通往那个交通工具的附近地区。
2. *Collective Transit Sampling:* 下一个函数对每个示例执行固定的次数。它可以进入所有过境顶点的组合邻域。

*随机游走*

抽样方法支持一下的随机游走

*多维随机游走*

1. 首先，从根顶点集中选择一个过境顶点。
2. 然后，将过境顶点的一个邻居添加到样本中，并替换根集中的过境顶点

*k-Hop邻居采样*

*层采样*

在每个步骤𝑖，层采样从样本的所有传输顶点的邻居集中采样𝑚𝑖顶点，直到样本大小达到用户给出的最大大小（𝑀）。

**使用NextDoor进行图采样**

*编程API*

input graph 初始的采样集合 用户自定义函数

stepTransits函数在给定步骤中返回样本的transit顶点

**GPU上的图采样的示例**

*采样并行性*

![image-20210506200818529](C:\Users\11957\AppData\Roaming\Typora\typora-user-images\image-20210506200818529.png)



* **Individual Transit Sampling**

  在每一步𝑖中，将连续的$m_i$线程分配给一对样本和transit。每个线程在它的transit上调用用户定义函数。

* **Collective Transit Sampling**

首先计算全部transit 的全部邻节点

* **Limitations**

尽管API启用了更细粒度的方法，但采样并行性无法使用GPU，原因如下。

1. 在 individual transit sampling 中对于每个样本，算法并行调用几个transit顶点的邻居。有的线程处理的节点邻居少，会出现warp分支的情况。
2. 该算法的负载平衡能力也很差，原因是transit节点的邻居数量不均。
3. 图必须存储在全局内存中，因此访问过境顶点的邻居会导致高延迟

*Transit并行性*

Transit并行性将具有相同transit vertex的所有样本进行分组，并通过将这些样本分配给连续线程来处理一个transit vertex的所有样本。

1. 在GPU上运行采样之前，我们通过将与同一transit vertex相关的所有样本分组，创建transit vertex到其样本的映射
2. 我们将每个transit vertex分配给一组线程，它们可以组织为grid、thread block或warp。



- 在individual transit sampling 中

将每个sample分配给组中连续的线程，每个线程调用next将transit的一个邻居添加到它的sample中

- 在 collective transit sampling 中

我们通过将每个样本分配给transit group（grid,thread block, or warp）中的连续线程，来创建传输的组合邻域，并且组中的连续线程将传输的邻域添加到样本的组合邻域。

优点：

* 连续线程执行类似的工作，因为每个线程用相同的邻居调用下一个函数
* 连续的线程可以访问同一传输工具的邻居。

 **Efficient Transit Parallelism on GPUs**

 **6.1   Sampling in Individual Transit Sampling**

- NextDoor使用了三种级别的并行性：transits到thread block、sample到warp，以及next函数的执行到一个 thread

**Sub-warps**

在理想的情况下，warp和sample之间将存在一对一的关系，这将确保warp中的每个线程，使用一个合并事务写入全局内存。

但是，每个warp都有固定数量的线程数（通常是32），这个数字有时可能大于next函数所需的执行次数

NEXTDoor在几个samples中共享一个warp

*sub-warp*指的是分配给相同sample的一组连续线程，NextDoor用sub-warp来作为基本的资源调度单元。

具有相同的size

**Load balancing**

要解决的问题：

1、因此，如果总是为每个transit 顶点分配一个线程块，我们就会得到次优的性能。在此限制下，transit顶点所需的线程数可能超过块中线程数的限制。

（因为在三级并行下，transit顶点一步需要与将添加到样本的邻居总数一样多的线程）

2、另一方面，如果一个transit顶点只需要少量的线程，则将整个线程块专用给传输，就会浪费GPU资源

因此NEXTDOOR 使用了三种类型的kernel

1. sub-warp核在一个单个warp中处理几个transit顶点。

   它仅适用于需要比warp（size=32）更小的线程的transit顶点

2. thread block内核将一个线程块给单个传输顶点，

   它仅适用于需要比warp中更多线程但小于最大thread block大小（1,024）的传输顶点。

3. grid kernel在几个thread block中处理单个transit顶点

   它只适用于过境顶点需要超过1,024个线程。

**调度**

要将transits分配给kernels，NextDoor将为每个传输顶点创建一个调度索引。包含三个阶段。

1. 首先，NextDoor根据从*stepTransits*函数获得的transits创建一个 transit-to-sample的映射

2. 然后，NextDoor可使用并行扫描操作，根据与每个transits顶点相关联的样本数量，将所有transits顶点划分为三组。

3. 最后，一个transit顶点的调度索引被设置为其集中的transit顶点的索引。

   在为transit顶点选择一个kernel之后，我们根据线程索引将transit顶点的每个样本分配给内核中的一个sub-warp。

**Catching**

NextDoor对不同的内核使用不同的缓存策略，以最小化内存访问成本。

在grid和thread block内核中的transit顶点邻居采样时，这些内核的线程块将transit顶点的邻居加载到shared memory中。

当邻居数目不适合于shared memory时，NextDoor会从global memory中加载邻居。

warp shuffle 指令：允许连续的线程从彼此的寄存器中读取邻居。

**6.2 Transit-Parallel Collective Transit Sampling**

Collective transit sampling 应用需要计算每个样本的所有transits的组合邻接点。

这是一个潜在的性能瓶颈，因此NextDoor使用传输并行性来加快进程。

它构建组合的邻域，就像一个只运行一步的 transit采样应用程序一样。

NextDoor没有从一个 transit点的邻域中采样新的顶点，而是将邻域中的所有顶点添加到样本的组合邻域中。

在每个样本建立一个单一的组合交通邻域后，原则上可以检测哪些样本具有相同的组合邻域，并以横向平行的方式扩展所有这些样本

**6.3 Unique Neighbors**

对于某些应用程序，要求来自所有transit 顶点的所有采样邻居都是唯一的

在每一个步骤采样后，NextDoor删除重复的采样顶点，

首先使用并行基数排序对它们进行排序，然后使用并行扫描获得不同的顶点。

如果采样的邻居适合共享内存，那么NextDoor通过将一个样本分配给一个线程块来执行此计算，否则将为每个样本调用一个内核

在此过程之后，如果样本大小小于步长大小，则NextDoor将使用采样并行方法而不是传输并行方法来执行采样。

**6.4 Graph Sampling using Multiple GPUs**

首先，NextDoor在所有GPU中平均分配样本。然后，NextDoor执行负载平衡和调度，并独立地在每个GPU上调用采样内核。最后，它收集来自所有GPU的输出。

**6.6 Advantages of NextDoor’s API**



它以细粒度的方式描述采样操作，从而通过样本和传输并行性更有效地使用GPU的执行层次结构。

NextDoor的API是通用的，支持不同类型的图采样应用程序，而现有的系统要么运行特定的图采样应用程序

或

只支持表示特定类型的图采样应用程序，例如Random walk·

此外，NextDoor的API要求应用程序明确指示每个示例中的哪些顶点是过transit顶点。

最后，API要求应用程序显式地说明在每个步骤中必须采样的顶点的数量。NextDoor使用此信息跨GPU的执行层次有效负载平衡计算。

**7 Alternative Graph Processing Systems**

基于GPU的图处理系统可以分成两类。

- 消息传递
- Frontier Centric

**Message-passing Abstraction**

利用消息传递实现的图采样的 transit-parallel方法如下。

- 首先，在与transit相关的每个样本的每一步中，对transit的邻居进行采样。
- 然后，调用stepTransits函数来检索下一步的传输，相关样本以消息的形式发送到传输。
- 每个transit vertex只与一个线程关联，该线程按顺序处理其所有样本。

**Frontier-centric Abstraction**

它利用了在图计算的任何步骤之后，生成一组边界顶点的属性

高级运算符包含用户定义的采样条件，它在传输顶点的每个邻居上调用

此操作将决定是否应将邻居添加到样本中。在这种情况下，调用stepTransits来检索样本的transit顶点，并将transit顶点添加到新的边界中。

**Limitations over NextDoor**

首先，这些系统只考虑一个程度的并行性，即所有的transit顶点都可以并行处理，但每个过境的样本都可以按顺序处理

其次，这些系统基于每个顶点的邻居的数量来平衡负载，因为在传统的图处理应用程序中访问了每个顶点的所有邻居
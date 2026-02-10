# ATask_Runner

提供一个线程完全的管线，用于并行处理任务。

## 获取模型：
将模型复制到 ./model 目录下

    cd ./model
    rsync -azP /media/nas/work/model/ .

模型可以从[百度云盘](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)下载，里面包含了行为，人脸相关模型，也包含了 [BiCifParaformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 模型导出的 enccode, decode, predictor, stamp 模型等等 ...

## 生成模型配置模板

``` text
python test.py --build_model_config
```
将在 ./config_temp 目录下生成每个模型的配置文件，将其复制到 ./config 目录下，并根据实际情况修改配置文件。

注意：其中 APipe.yaml 与众不同，包含 APipe 的默认配置

## 设计思路

**将AI任务抽象为 ATask**，每个 ATask 包含：
- 做什么：使用模型"位或"声明要执行的模型任务，如 DO_ACT | DO_FACEDET 意味着要执行”行为检测“ 和 ”人脸检测“
- 输入数据：如图像，或声音；
- 配置参数：各种属性，阈值等；
- 输出数据：存储结果，中间结果；

**通过先进先出队列和执行体并行处理 ATask**：
- 三个执行体：执行体本质是一个线程池，从输入队列读取 ATask，执行任务，填充结果，写入输出队列
	- E_PRE：预处理
	- E_INFER：推理器，一般是 onnxruntime，ascend om 执行体；
	- E_POST：后处理，执行完成后，修改 ATask.todo，判断是否所有任务都已经完成（todo\=\=0)，如果都完成，设置 finished，否则扔到 Q_SUB_INP 中继续下个任务的执行；
- 五个队列：
	- Q_INP：输入队列，使用者能直接访问的队列，构造 ATask 后，扔到 Q_INP 队列；
	- Q_PRE：预处理完队列，存储执行完预处理的任务
	- Q_INFER：推理完队列，存储推理完成的任务
	- Q_RESULT：后处理完队列，存储后处理完成的任务
    - Q_SUB_INP: 当 ATask 为组合任务时，存储未完成的子任务，E_PRE 会优先从 E_SUB_INP 获取；
	
通过**统计四个队列的空闲状态，能够判断执行瓶颈**，调整三个执行体的优先级，线程数达到最高效率，一般来说 Q_PRE 缓冲空，则说明数据提供的太慢了；

**任务类型**：任务之间是互相关联的，如人脸识别依赖人脸检测结果，分成两类任务：
1. 检测任务：对整图处理，输出子任务需要的数据，如人脸框的位置，如 VAD 切片位置等；
2. 子任务：根据检测任务输出，从整图中抠图后再推理，子任务之间也有顺序，如先进行人脸质量打分，再决定是否执行人脸识别

**ATask 的规模是需要认真考虑的**，如 asr 转写任务，需要：
1. 输入整节课的声音数据
2. VAD 切片
3. 对每个 VAD 切片做 ASR
则应该将整节课的声音作为一个 ATask 呢，还是将 VAD 切片后的每个声音片段作为一个 ATask 呢？理论上每个 ATask 越小，并行效率越高，但会打乱目前的流程：如使用每片声音作为一个 ATask，会出现多杰课片段混合在一起执行的情况；执行完成需要合并。

模型实例问题：理论上，可以使用单实例的性价比最高（内存，gpu内存），但若模型每层计算量太小，无法重复利用硬件资源，因此应考虑模型池；


## 数据结构

##### ATask
ATask 承载一个最小的数据处理单元，在管道中传递，预处理/推理/后处理函数将结果数据存储到 data 中。一个 ATask 同时只能被一个线程拥有并访问；

``` python
class ATask:
	def __init__(self):
		self.todo = 0            ## 通过位或指定要执行的操作
		self.inpdata = None      ## 只读：输入数据，可能是图像或声音
		self.data = {}           ## 读些：数据字典，包含预处理结果，推理中间数据，后处理结果等
		self.userdata = {}       ## 只读：各种配置参数
		self.finished = False    ## 只读：当 finished 时，说明所有任务执行完成，从 data 中可以安全的读取结果数据
		
```

##### AFifo
AFifo 是一个线程安全的先进先出队列，可以直接使用 python3 的 class queue，一般来说只有 Q_INP 需要限制排队上限，防止OOM；

##### AModel
具体模型的功能实现，包含预处理/推理/后处理，所有操作都应对 task 进行；

必须保证：
1. 每个函数必须闭包！！！
2. 每个函数必须可重入！！！
具体实现必须使用线程局部存储；

###### 已经确认：<font color='green'>onnxruntime, ascend om，tensorRT 等推理库均支持“重入”，即单实例支持跨线程推理；</font>

``` python
class AModel:
	def preprocess(self, task:ATask): ...
	def infer(self, task:ATask): ...
	def postprocess(self, task:ATask): ...
```

##### AModelBackend
每个 AModel 必须绑定对应的 backend，目前仅仅支持 onnxruntime。
一个 AModel 可能同时绑定多个 backend，根据模型配置，创建 backend 实例。

``` python
class AModelBackend:
    def setup(self, model_path:str, **kwargs):
		## 实例化后端，根据 kwargs 配置，该配置来自 config/model.yaml 的 backend_cfg 部分
        pass

    def teardown(self):
		## 结束
        pass

    def get_input_num(self) -> int:
        ## 返回模型的 input tensor 个数
        return -1
    
    def get_input_shape(self, idx:int) -> tuple[int, ...]:
        ## 返回第 idx 个 input tensor 的 shape
        return (-1,)
    
    def get_input_dtype(self, idx:int) -> str:
        ## 返回第 idx 个 input tensor 的 dtype
        return ""

    def get_output_num(self) -> int:
        ## 返回模型的 output tensor 个数
        return -1
    
    def get_output_shape(self, idx:int) -> tuple[int, ...]:
        ## 返回第 idx 个 output tensor 的 shape
        return (-1,)
    
    def get_output_dtype(self, idx:int) -> str:
        ## 返回第 idx 个 output tensor 的 dtype
        return ""    

    def infer(self, inputs:Tuple[Any]) -> List[Any]:
        ## 执行推理，输入 inputs 为一个列表，每个元素都是对应的 input tensor
        ## 输出 outputs 也是一个列表，每个元素都是对应的 output tensor
		## 派生类必须实现
        raise NotImplementedError("infer not implemented")
```

##### AExecutor
执行体，本质是一个线程池，从输入队列获取下一个 ATask，根据 ATask.todo 找到 AModel 实例，调用相应的处理；
特别的，E_POST 需要额外处理 ATask.todo：
- E_POST 调用 AModel.postprocess() 之后，需要将 ATask.todo 中当前执行的任务的“位”清空，如当前任务是 DO_ACT，则 ATask.todo &= ~DO_ACT；
- 若 ATask.todo == 0 说明所有任务都执行完成，设置 ATask.finished = True，否则将 ATask 再次扔到 Q_INP 中，等待下一轮处理；

``` python
class AExecutor:
	def __init__(self, in_queue, out_queue, thread_num=1):  ## 初始化输入，输出队列，工作线程数
		self.in_queue = in_queue
		self.out_queue = out_queue
		self.work_threads = [
			threading.Thread(target=self.run) for _ in range(thread_num)
		]
		
	def __call__(self, model:AModel, task:ATask): ...           ## 调用具体实例的 pre/infer/post 实现
	
	def run(self):                                    ## 工作线程
		while 1:                                      ## 线程池循环
			task = self.in_queue.get()                ## 从输入队列获取下一个 ATask
			model = get_model_instance(task.todo)     ## 根据 todo 找到 model 实例
			self(model, task)                         ## 执行具体实现
			self.out_queue.put(task)                  ## 扔到输出队列
```

##### APipe
管线，根据模型配置加载需要的模型，构造模型依赖关系；
实例化 AExecutor, AFifo, 并创建管道拓扑；
接受 ATask 执行，并返回结果；

``` python
class APipe:
    def post_task(task:ATask): ...          ## 投递 ATask 到 Q_INP
    def wait(self) -> ATask: ...            ## 等待下一个完成的任务，注意，不一定按照 post 的顺序返回!!!
    def get_qsize(self) -> Tuple[int, int, int, int]: ...  ## 返回四个 queue 的等待数，一定程度上可以用于评估性能瓶颈
```

## 测试

    python test_image.py
    python test.py --test_all
''' APipe 对应一个完整管道，

        包含 E_pre + E_infer + E_post + Q_inp + Q_pre + Q_infer + Q_post


    典型使用流程：

        pipe = APipe()
    
        ## 数据采集线程
        while 1:
            pcm = next_pcm()    ## 获取声音
            pic = next_pic()    ## 获取图像

            task_pic = ATask(DO_ACT | DO_FACEDET ..., pic, userdata={})
            pipe.post_task(task_pic)

            task_pcm = ATask(DO_ASR_XXX ..., pcm, userdata={})
            pipe.post_task(task_pcm)


        ## 结果处理线程
        while 1:
            task = pipe.wait()
            task.data 中存储推理结果
            ....
            
'''

from .atask import ATask
from .aexecutor import AExecutor
from .model_desc import load_model_config
from .amodel import AModel
from .model_id import DO_ASR_VAD
from typing import Tuple
from queue import Queue
import logging

logger = logging.getLogger("pipe")

class APipe:
    def __init__(
        self,
        Q_inp_size: int=4,
        E_pre_num_thread: int=2,
        E_infer_num_thread: int=2,
        E_post_num_thread: int=2,
        model_mask: int=-1,         ## 启用的模型 bitmask，默认启用所有
        model_config_path: str="./config",
        debug: bool=False,
        debug_one_thread: bool=False, ## 如果启动单线程，则顺序执行
    ):
        self.debug = debug
        self.debug_one_thread = debug_one_thread

        self.__supported_todo = 0
        self.__model_desc = load_model_config(model_config_path, model_mask)
        if not self.__model_desc:
            raise Exception("no valid model found")
        
        for cfg in self.__model_desc:
            self.__supported_todo |= cfg["mid"] ## 所有支持的 todo 位

        ## 队列
        self.Q_inp = Queue(maxsize=Q_inp_size)
        self.Q_inp_sub = Queue()
        self.Q_pre = Queue()
        self.Q_infer = Queue()
        self.Q_result = Queue()     ## 所有 todo 均完成的 task 队列

        ## 执行器
        def find_model(task:ATask) -> AModel | None:
            return self.__get_model_from_todo(task)
        
        self.E_pre = AExecutor(
            "pre", find_model, 
            self.Q_inp, self.Q_pre, 
            E_pre_num_thread,
            sub_q = self.Q_inp_sub,
        )
        self.E_infer = AExecutor(
            "infer", find_model, 
            self.Q_pre, self.Q_infer, 
            E_infer_num_thread
        )
        self.E_post = AExecutor(
            "post", find_model, 
            self.Q_infer, self.Q_inp_sub, 
            E_post_num_thread, 
            result_q=self.Q_result
        )

        ## 根据配置的模型，加载所有模型
        self.__models = self.__load_models()    # [ (mid, instance), ... ]

    def close(self):
        self.E_infer.close()
        self.E_pre.close()
        self.E_post.close()

    def __repr__(self):
        info = f"<APipe> ins={id(self)}, enable model:{self.__supported_todo:b}\n"
        info += f"    Q_inp:{self.Q_inp.qsize()}, Q_pre:{self.Q_pre.qsize()}, Q_infer:{self.Q_infer.qsize()}, Q_result:{self.Q_result.qsize()}, Q_sub:{self.Q_inp_sub.qsize()}\n"
        return info
        
    def post_task(self, task:ATask):
        ''' 投递任务到 E_inp
            XXX: 如果 E_inp 满，将阻塞，
            总是返回成功
        '''
        if task.todo & ~self.__supported_todo:
            logger.error(f"unsupported todo {task.todo:b} vs {self.__supported_todo:b}")
            raise Exception("unsupported todo")
        
        self.Q_inp.put(task)

    def wait(self) -> ATask:
        '''
        等待下个完成的任务，注意，不一定按照 post 的顺序返回!!!
        '''
        return self.Q_result.get()

    def get_qsize(self) -> Tuple[int, int, int, int]:
        ## 返回四个 queue 的等待数，一定程度上可以用于评估性能瓶颈
        return self.Q_inp.qsize(), self.Q_pre.qsize(), self.Q_infer.qsize(), self.Q_result.qsize()
    
    def __get_model_from_todo(self, task:ATask) -> AModel | None:
        ## 被执行器调用，根据模型顺序，以及 task 剩余的 todo 位，找到对应的模型
        ## 如果找不到，则返回 None
        for mid, model_run in self.__models:
            if mid & task.curr_todo():
                return model_run
        return None
            
    def __load_models(self):
        # TODO: to load enabled models, and do register
        import importlib
        models = [
            # (mid, instance)
        ]
        for cfg in self.__model_desc:
            name = cfg['name']
            mid = cfg['mid']
            if mid == DO_ASR_VAD:
                logger.warning(f"load_models: ignore DO_ASR_VAD")
                continue
            
            ## 从 f"models/{name}.py" 中 import f"Model_name" 类，并实例化
            logger.info(f"APipe: load_models: load {name} from ./models.{name}")
            module = importlib.import_module(f".models.{name}", package=__package__)
            cls = getattr(module, f"Model_{name}")
            model = cls(**cfg)
            logger.info(f"APipe: load_models: load {name} instance={model}")
            models.append((mid, model))

        return models


class APipeWrap:
    def __init__(self, model_mask: int=-1):
        self.__model_mask = model_mask

    def __enter__(self):
        self.__pipe = APipe(model_mask=self.__model_mask)
        return self.__pipe

    def __exit__(self, exc_type, exc_value, traceback):
        self.__pipe.close()

import threading
from queue import Queue, Empty
from .atask import ATask, ATask_Quit
from .amodel import AModel
import logging
from typing import cast
from .timeused import TimeUsed

logger = logging.getLogger("executor")

class AExecutor:
    def __init__(
        self, 
        name: str,      ## "pre", "infer", "post"
        func_find_model,    ## APipe 提供，根据 ATask 找到对应的 AModel
        inp_q: Queue, 
        out_q: Queue, 
        num_thread: int=1,
        sub_q=None,     ## 当 name="pre" 时，优先从该 queue 获取
        result_q=None,  ## 当 name="post" 时需要，存储结果
        debug: bool=False, ## 如果启动单线程，则顺序执行
        profile: bool=False,    ## 是否启用性能统计
    ):
        self.debug = debug
        self.profile = profile
        if name == "post" and not isinstance(result_q, Queue):
            raise TypeError("result_q must be a Queue when name=='post'")
        
        if name not in ("pre", "infer", "post", "once"):
            raise ValueError(f"unknown executor name {name}, should be one of ('pre', 'infer', 'post)")
        
        if not callable(func_find_model):
            raise TypeError("func_find_model must be callable")

        self.name = name
        self.__func_find_model = func_find_model
        self.inp_q = inp_q
        self.out_q = out_q
        self.res_q = cast(Queue, result_q)
        self.sub_q = cast(Queue, sub_q)
        run_proc = self.__run
        if profile:
            run_proc = self.__run_with_profile
        self.ths = [
            threading.Thread(target=run_proc, name=f"{name}_{i}") for i in range(num_thread)
        ]
        for th in self.ths: th.start()

    def close(self):
        logger.debug(f"{self.__class__.__name__}.exit {self.name}, threads:{len(self.ths)}")
        for _ in self.ths:
            self.inp_q.put(ATask_Quit())
        for th in self.ths: th.join()

    def __run(self):
        def next_pre_task() -> ATask:
            while 1:
                try:
                    return self.sub_q.get_nowait()
                except Empty:
                    try:
                        return self.inp_q.get(timeout=0.1)
                    except Empty:
                        continue
            raise 
                
        while 1:
            if self.name == "pre":
                task = next_pre_task()
            else:
                task = self.inp_q.get()

            if not isinstance(task, ATask):
                logger.warning(f"{self.__class__.__name__} got non-ATask")
                break

            if isinstance(task, ATask_Quit):
                break

            todo = task.curr_todo()
            model = self.__func_find_model(task)
            if not isinstance(model, AModel):
                logger.warning(f"{self.__class__.__name__} cannot find AModel for todo:{todo:08x}")
                break

            if self.name == "pre":
                model._preprocess(task)
                self.out_q.put(task)
            elif self.name == "infer":
                model._infer(task)
                self.out_q.put(task)
            elif self.name == "post":
                model._postprocess(task)
                if not task.done(model.mid()):
                    self.out_q.put(task)
                else:
                    self.res_q.put(task)
            else:
                raise ValueError(f"unknown executor type {self.name}")
        return None

    def __run_with_profile(self):
        def next_pre_task() -> ATask:
            while 1:
                try:
                    return self.sub_q.get_nowait()
                except Empty:
                    try:
                        return self.inp_q.get(timeout=0.1)
                    except Empty:
                        continue
            raise 
                
        while 1:
            if self.name == "pre":
                task = next_pre_task()
            else:
                task = self.inp_q.get()

            if not isinstance(task, ATask):
                logger.warning(f"{self.__class__.__name__} got non-ATask")
                break

            if isinstance(task, ATask_Quit):
                break

            todo = task.curr_todo()
            model = self.__func_find_model(task)
            if not isinstance(model, AModel):
                logger.warning(f"{self.__class__.__name__} cannot find AModel for todo:{todo:08x}")
                break

            if self.name == "pre":
                with TimeUsed(f"pre: {model.name}"):
                    model._preprocess(task)
                self.out_q.put(task)
            elif self.name == "infer":
                with TimeUsed(f"infer: {model.name}"):
                    model._infer(task)
                self.out_q.put(task)
            elif self.name == "post":
                with TimeUsed(f"post: {model.name}"):
                    model._postprocess(task)
                if not task.done(model.mid()):
                    self.out_q.put(task)
                else:
                    self.res_q.put(task)
            else:
                raise ValueError(f"unknown executor type {self.name}")
        return None

'''
    atask 为一个最小任务，存储：
        做什么：todo, 通过位或指示需要进行的操作
        输入数据：inpdata, 格式为 np.ndarray，一般为图像，声音等
        配置或额外数据：userdata, dict
        输出数据，中间数据：data, dict
'''

import numpy as np
import threading
import logging
from .model_id import todo2str

# logger = logging.getLogger("asr_runner")

class ATask:
    def __init__(self, todo:int, inpdata:np.ndarray | tuple, userdata:dict):
        self.todo = todo
        self.inpdata = inpdata      ## indata 可以是单个数据，也可以是 tuple, 如 (batch_pcm, batch_samples)
        self.userdata = userdata
        self.data = dict()
        
        ## 以下变量内部使用
        self.__finished = False
        self.__todo = todo
        self.__lock = threading.Lock()
        self.__cv = threading.Condition(self.__lock)

    def __repr__(self):
        return f"todo=[{todo2str(self.todo)}], finished={self.__finished}"

    def wait(self):
        ''' 阻塞直到任务完成
        '''
        with self.__lock:
            while not self.__finished:
                self.__cv.wait()
        return self

    def done(self, curr_todo:int):
        ''' 当前任务完成时调用，会检查是否所有 todo 都完成了
            如果都完成，将设置 self.__finished 并通知 wait() 返回
            返回是否所有 todo 都完成了
        '''
        with self.__lock:
            self.__todo &= ~curr_todo
            if self.__todo == 0:
                self.__finished = True
                self.__cv.notify_all()
            return self.__finished
        
    def curr_todo(self):
        ''' 返回当前未完成的任务
        '''
        with self.__lock:
            return self.__todo
        

class ATask_Quit(ATask):
    def __init__(self):
        ## 用于结束管道
        super().__init__(0, np.empty((0,)), userdata={})
        self.__finished = True

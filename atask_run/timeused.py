import time
import threading
import psutil, os, sys
import pynvml

__all__ = ["TimeUsed", "TimeUsedSum"]

def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

class TimeUsedItem:
    def __init__(self, name):
        self.name = name
        self.count = 0              ## 统计次数
        self.total_duration = 0.0   ## 累积时长
        self.max_duration = 0.0     ## 最大时长
        self.min_duration = 999999999.999     ## 最小时长
        self.cpu_sum = 0.0
        self.cpu_count:int = 0
        self.cuda_mem_max = 0.0
        self.sys_mem_max = 0.0

    def __repr__(self) -> str:
        if self.cpu_count == 0:
            if self.count == 1:
                return f"{self.name}: count: {self.count}, " \
                    f"total: {self.total_duration:03f} secs, "\
                    f"avg:{self.total_duration/self.count:.06f} secs, "\
                    f"sys mem max:{self.sys_mem_max:.02f}M." \
                    f"cuda mem max:{self.cuda_mem_max:.02f}M."
            else:
                return f"{self.name}: count: {self.count}, " \
                    f"total: {self.total_duration:03f} secs, "\
                    f"avg:{self.total_duration/self.count:.06f} secs, "\
                    f"max:{self.max_duration:.05f} secs, "\
                    f"min:{self.min_duration:.05f} secs, "\
                    f"sys mem max:{self.sys_mem_max:.02f}M." \
                    f"cuda mem max:{self.cuda_mem_max:.02f}M."
        else:
            if self.count == 1:                
                return f"{self.name}: count: {self.count}, " \
                    f"total: {self.total_duration:03f} secs, "\
                    f"avg:{self.total_duration/self.count:.06f} secs, "\
                    f"sys mem max:{self.sys_mem_max:.02f}M, "\
                    f"cuda mem max:{self.cuda_mem_max:.02f}M, "\
                    f"avg cpu:{self.cpu_sum/self.cpu_count:.02f}%"
            else:
                return f"{self.name}: count: {self.count}, " \
                    f"total: {self.total_duration:03f} secs, "\
                    f"avg:{self.total_duration/self.count:.06f} secs, "\
                    f"max:{self.max_duration:.05f} secs, "\
                    f"min:{self.min_duration:.05f} secs, "\
                    f"sys mem max:{self.sys_mem_max:.02f}M, "\
                    f"cuda mem max:{self.cuda_mem_max:.02f}M, "\
                    f"avg cpu:{self.cpu_sum/self.cpu_count:.02f}%"

@singleton
class TimeUsedSum:
    def __init__(self):
        self.__lock = threading.Lock()
        self.items = {}   ## {<item_name> : <TimeUsedItem>}

    def __del__(self):
        self.dump()

    def update(
        self, 
        name:str, 
        duration: float, 
        cpu_sum:float=0.0, cpu_count:int=0,
        cuda_mem_max:float=0.0,
        sys_mem_max:float=0.0,
    ):
        with self.__lock:
            item = self.items.get(name, TimeUsedItem(name))
            item.count += 1
            item.total_duration += duration
            item.max_duration = max(item.max_duration, duration)
            item.min_duration = min(item.min_duration, duration)
            item.cpu_sum += cpu_sum
            item.cpu_count += cpu_count
            item.cuda_mem_max = cuda_mem_max
            item.sys_mem_max = sys_mem_max
            self.items[item.name] = item

    def dump(self):
        print("================= there are %d items =============="%len(self.items))
        for _,v in self.items.items():
            print(f"  {v}")

class TimeUsed:
    def __init__(self, name, with_cpu:bool=False, with_cuda:bool=False, cuda_device_id:int=0):
        self.name = name
        self.with_cpu = with_cpu
        self.with_cuda = with_cuda
        self.__cpu_sum = 0.0
        self.__cpu_count = 0
        self.__interval = 1.0
        self.__stop_event = threading.Event()
        self.__cuda_device_id = 0
        self.__cuda_mem_max:float = 0.0
        self.__sys_mem_max:float = 0.0

    def __cpu_sampler(self):    
        p = psutil.Process(os.getpid())
        p.cpu_percent(interval=None)
        while not self.__stop_event.wait(self.__interval):
            usage = p.cpu_percent(interval=self.__interval)
            self.__cpu_sum += usage
            self.__cpu_count += 1
            mem = p.memory_info()
            self.__sys_mem_max = max(self.__sys_mem_max, mem.rss)

    def __cuda_sampler(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.__cuda_device_id)  # 默认 GPU 0
        while not self.__stop_event.wait(self.__interval):
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            ## 存储 cuda 最大显存使用量
            if hasattr(meminfo, 'used'):
                self.__cuda_mem_max = max(self.__cuda_mem_max, float(meminfo.used))
        pynvml.nvmlShutdown()

    def __enter__(self):
        if self.with_cpu:
            self.__cpu_sum = 0.0
            self.__cpu_count = 0
            self.__cpu_th = threading.Thread(target=self.__cpu_sampler)
            self.__cpu_th.start()
        if self.with_cuda:
            self.__cuda_th = threading.Thread(target=self.__cuda_sampler)
            self.__cuda_th.start()
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_time = time.perf_counter()
        duration = self.stop_time - self.start_time

        if self.with_cpu:
            self.__stop_event.set()
            self.__cpu_th.join()
        if self.with_cuda:
            self.__stop_event.set()
            self.__cuda_th.join()

        TimeUsedSum().update(
            self.name, 
            duration, 
            cpu_sum=self.__cpu_sum, cpu_count=self.__cpu_count,
            cuda_mem_max=self.__cuda_mem_max / 1024 / 1024,
            sys_mem_max=self.__sys_mem_max / 1024 / 1024,
        )

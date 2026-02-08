import time

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

    def __repr__(self) -> str:
        return f"{self.name}: count: {self.count}, " \
            f"total: {self.total_duration:03f} seconds, "\
            f"avg:{self.total_duration/self.count:.06f} seconds, "\
            f"max:{self.max_duration:.05f} second, "\
            f"min:{self.min_duration:.05f} seconds"

@singleton
class TimeUsedSum:
    def __init__(self):
        self.items = {}   ## {<item_name> : <TimeUsedItem>}

    def __del__(self):
        self.dump()

    def update(self, name:str, duration: float):
        item = self.items.get(name, TimeUsedItem(name))
        item.count += 1
        item.total_duration += duration
        item.max_duration = max(item.max_duration, duration)
        item.min_duration = min(item.min_duration, duration)
        self.items[item.name] = item

    def dump(self):
        print("================= there are %d items =============="%len(self.items))
        for _,v in self.items.items():
            print(f"  {v}")

class TimeUsed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_time = time.perf_counter()
        duration = self.stop_time - self.start_time
        TimeUsedSum().update(self.name, duration)

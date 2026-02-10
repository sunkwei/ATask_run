from ..atask import ATask
from ..amodel import AModel
import numpy as np
from ..numpy_fbank import fbank
from .v4_preprocess import V4Preprocess

pre_process = V4Preprocess()
titles = ['discuss', 'noise', 'read-aloud', 'single']
class Model_v4(AModel):
    def _preprocess(self, task: ATask):

        v4_pre = pre_process(task.inpdata)
        task.data["v4_pre"] = v4_pre

        if len(v4_pre) == 0:
            task.data["td"] = []
        else:
            sec = len(task.inpdata) // 16000
            if sec == 0:
                td = [round(len(task.inpdata) / 16000, 2)]
                task.data["td"] = td
            else:
                td = [1] * sec
                during = len(task.inpdata) / 16000
                if round(during - sec, 2) < 0.2:
                    td[-1] = round(td[-1] + (during - np.float32(sec)), 2)
                else:
                    td.append(round(during - np.float32(sec), 2))
                task.data["td"] = td

    def _infer(self, task: ATask):
        assert "v4_pre" in task.data, f"'v4_pre' not found in task.data, {task.data.keys()}"
        if len(task.data["v4_pre"]) == 0:
            task.data["v4_infer"] = []
            task.data["td"] = []
        
        else:
            r = self(task.data["v4_pre"])[0]
            out = np.array(titles)[np.argmax(r,axis=1)]
            r = out.tolist()

            if len(r) == 0:
                task.data["v4_infer"] = ["noise"]
                task.data["td"] = [len(task.inpdata) / 16000]

            elif len(r) > len(task.data["td"]):
                task.data["v4_infer"] = r[:len(task.data["td"])]

            else:
                task.data["v4_infer"] = r + ["noise"] * (len(task.data["td"]) - len(r))

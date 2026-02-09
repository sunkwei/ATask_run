from ..atask import ATask
from ..amodel import AModel
import numpy as np
from ..numpy_fbank import fbank

class Model_sv(AModel):
    def _preprocess(self, task: ATask):

        pcm = task.inpdata
        wav_part = pcm#pcm[:int(3 * 16000)]
        feat = fbank(wav_part)
        feat = np.array([feat])
        task.data["sv_pre"] = feat

    def _infer(self, task: ATask):
        assert "sv_pre" in task.data, f"'sv_pre' not found in task.data, {task.data.keys()}"
        task.data["sv_infer"] = self(task.data["sv_pre"][None])[0]
    
    def _postprocess(self, task: ATask):
        assert "sv_infer" in task.data and "sv_infer" in task.data
        d = np.linalg.norm(task.data["sv_infer"], axis=1, keepdims=True)
        f = task.data["sv_infer"] / d     
        task.data["sv_feat"] = f
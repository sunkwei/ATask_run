from ..atask import ATask
from ..amodel import AModel
from .asr_frontend import WavFrontend
import os.path as osp
import yaml
import numpy as np

class Model_asr_enc(AModel):
    def preprocess(self, task: ATask):
        if not hasattr(self, "frontend"):
            model_path = osp.join(osp.dirname(self.model_path()), "asr_enc")
            if not osp.exists(model_path):
                raise FileNotFoundError(f"{model_path} is not found.")
            with open(osp.join(model_path, "config.yaml")) as f:
               config = yaml.load(f, Loader=yaml.FullLoader)
            self.frontend = WavFrontend(
                cmvn_file=osp.join(model_path, "am.mvn"),
                **config["frontend_conf"]
            )
        ## task.inpdata 必须是 f32 的 pcm
        speech, _ = self.frontend.fbank(task.inpdata)
        speech, speech_len = self.frontend.lfr_cmvn(speech)
        mask3 = np.ones((1, speech_len, 1), dtype=np.float32)
        mask4 = np.ones((1, 1, 1, speech_len), dtype=np.float32)
        task.data["asr_enc_inp"] = (speech[None, ...], mask3, mask4)
    
    def infer(self, task: ATask):
        assert "asr_enc_inp" in task.data, f"'asr_enc_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_enc_infer"] = self.impl().infer(task.data["asr_enc_inp"])
    
    def postprocess(self, task: ATask):
        pass
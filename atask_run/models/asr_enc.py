from ..atask import ATask
from ..amodel import AModel
from .asr_frontend import WavFrontend
import os.path as osp
import yaml
import numpy as np
import logging 

logger = logging.getLogger("asr_enc")

class Model_asr_enc(AModel):
    def _preprocess(self, task: ATask):
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

        ## 如果批次模式，task.userdata.get("batch", False) == True
        ## task.inpdata 必须是 f32 的 pcm 也有可能是批次
        if isinstance(task.inpdata, np.ndarray):
            ## 单 pcm 模式
            assert task.userdata.get("batch", False) == False
            speech, _ = self.frontend.fbank(task.inpdata)
            speech, speech_len = self.frontend.lfr_cmvn(speech)
            mask3 = np.ones((1, speech_len, 1), dtype=np.float32)
            mask4 = np.ones((1, 1, 1, speech_len), dtype=np.float32)
            mask4 = (1 - mask4) * -10000.0
            task.data["asr_enc_inp"] = (speech[None, ...], mask3, mask4)
            task.data["asr_enc_mask"] = np.ones((1, 1, speech_len), dtype=np.float32)

        elif isinstance(task.inpdata, tuple):
            ## 批次模式，(batch_pcm, batch_samples)
            assert task.userdata.get("batch", False) == True
            speechs = []; mask3s = []; mask4s = []; enc_masks = []
            batch_pcm, batch_samples = task.inpdata

            for i in range(len(batch_pcm)):
                speech, _ = self.frontend.fbank(batch_pcm[i])
                speech, speech_len = self.frontend.lfr_cmvn(speech)
                mask3 = np.ones((1, speech_len, 1), dtype=np.float32)
                mask4 = np.ones((1, 1, 1, speech_len), dtype=np.float32)
                enc_mask = np.ones((1, 1, speech_len), dtype=np.float32)
                speechs.append(speech[None, ...]); mask3s.append(mask3); mask4s.append(mask4); enc_masks.append(enc_mask)

            max_len = max(s.shape[1] for s in speechs)
            for i in range(len(batch_pcm)):
                ## speech: (1, T, 560)
                pad_right = max_len - speechs[i].shape[1]
                speechs[i] = np.pad(speechs[i], ((0, 0), (0, pad_right), (0, 0)), mode='constant', constant_values=0)

                ## mask3: (1, T, 1)
                mask3s[i] = np.pad(mask3s[i], ((0, 0), (0, pad_right), (0, 0)), mode='constant', constant_values=0)

                ## mask4: (1, 1, 1, T)
                mask4s[i] = np.pad(mask4s[i], ((0, 0), (0, 0), (0, 0), (0, pad_right)), mode='constant', constant_values=0)

                ## enc_mask: (1, 1, T)
                enc_masks[i] = np.pad(enc_masks[i], ((0, 0), (0, 0), (0, pad_right)), mode='constant', constant_values=0)
            
            speech = np.concatenate(speechs, axis=0)
            mask3 = np.concatenate(mask3s, axis=0)
            mask4 = np.concatenate(mask4s, axis=0)
            mask4 = (1 - mask4) * -10000.0
            enc_mask = np.concatenate(enc_masks, axis=0)

            task.data["asr_enc_inp"] = (speech, mask3, mask4)
            task.data["asr_enc_mask"] = enc_mask

    def _infer(self, task: ATask):
        assert "asr_enc_inp" in task.data, f"'asr_enc_inp' not found in task.data, {task.data.keys()}"
        task.data["asr_enc_infer"] = self(task.data["asr_enc_inp"])

        if self.debug:
            asr_enc_infer = task.data["asr_enc_infer"][0]
            logger.debug(f"asr_enc data: {asr_enc_infer.shape}")
    
    def _postprocess(self, task: ATask):
        pass
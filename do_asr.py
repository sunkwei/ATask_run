import numpy as np
from typing import cast
import logging
import threading
from atask_run.models.asr_vad import Model_asr_vad
import atask_run.model_id as mid
from atask_run.apipe import APipe
from atask_run.atask import ATask

logger = logging.getLogger("asr runner")

class ASRRunner:
    def __init__(self, pipe=None, debug=False):
        self.debug = debug
        self.__cached_pcm = np.empty((0, ), np.float32)
        self.__stamp_ms_cached_pcm = 0
        self.__sample_once_vad = 6 * 16000
        self.__V = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        self.__mod_mask = mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP
        if isinstance(pipe, APipe):
            self.__P = cast(APipe, pipe)
            self.__owner = False
        else:
            self.__P = APipe(model_mask=self.__mod_mask)
            self.__owner = True
        logger.info(f"ASRRunner: mod_mask={self.__mod_mask:b}, pipe={self.__P}")

    def __del__(self):
        del self.__V
        if self.__owner:
            del self.__P
        logger.info(f"ASRRunner: del")
        
    def update_stream(self, pcm:np.ndarray, last:bool=False) -> list:
        '''
            适合每次写入一段

            循环，每次 vad 得到一条片段，执行 asr 得到结果，再 vad ...
        '''
        if self.debug:
            logger.info(f"{self.__class__.__name__}: update_stream: pcm.shape={pcm.shape}, last={last}")

        self.__cached_pcm = np.concatenate([self.__cached_pcm, pcm])
        head, tail = 0, len(self.__cached_pcm)
        asr_result = []
        last_stamp = self.__stamp_ms_cached_pcm
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(self.__cached_pcm[head : head + self.__sample_once_vad])
            if begin_ms >= 0:
                if begin_ms < end_ms:
                    rs = self.__do_asr(begin_ms, end_ms)
                    asr_result.append(rs)
                    if self.debug:
                        txt = ''.join(rs['tokens'])
                        logger.info(f"    GOT vad seg: {begin_ms}-{end_ms}, txt:{txt}, pcm:{rs['pcm']}")
                last_stamp = end_ms
            head += self.__sample_once_vad

        if last:
            if head < tail:
                pad_samples = self.__sample_once_vad - (tail - head)
                self.__cached_pcm = np.pad(self.__cached_pcm, (0, pad_samples), mode='constant', constant_values=(0))
                begin_ms, end_ms = self.__V.update_for_asr(self.__cached_pcm[head:])
                if begin_ms >= 0 and begin_ms < end_ms:
                    rs = self.__do_asr(begin_ms, end_ms)
                    asr_result.append(rs)
                    if self.debug:
                        txt = ''.join(rs['tokens'])
                        logger.info(f"    GOT vad seg: {begin_ms}-{end_ms}, txt:{txt}")

            begin_ms, end_ms = self.__V.update_for_asr(None)
            if begin_ms >= 0 and begin_ms < end_ms:
                rs = self.__do_asr(begin_ms, end_ms)
                asr_result.append(rs)
                if self.debug:
                    txt = ''.join(rs['tokens'])
                    logger.info(f"    GOT vad seg: {begin_ms}-{end_ms}, txt:{txt}")

            ## 所有清空
            self.__V.reset()
            self.__cached_pcm = np.empty((0, ), np.float32)
            self.__stamp_ms_cached_pcm = 0
        else:
            ## 删除不再需要的 pcm
            used_samples = 16 * (last_stamp - self.__stamp_ms_cached_pcm)
            self.__cached_pcm = self.__cached_pcm[used_samples:]
            self.__stamp_ms_cached_pcm = last_stamp

        return asr_result
    
    def update_file(self, pcm:np.ndarray) -> list:
        '''
            适合一次 asr 处理一节课
            先做 Vad 得到所有片段，然后扔到 APipe 中，启动一个工作线程，接受结果
        '''
        logger.info(
            f"{self.__class__.__name__}: update_file: pcm.shape={pcm.shape}, "
            f"duration:{len(pcm)/16000:.03f} seconds"
        )
        results = []

        def wait_proc(pipe:APipe, rs:list):
            while 1:
                task = pipe.wait()
                r = {
                    "begin_ms": task.userdata["begin_ms"],
                    "end_ms": task.userdata["end_ms"],
                    "tokens": task.data["asr_dec_token"],
                    "stamps": task.data["asr_dec_stamp"]
                }

                if self.debug:
                    r["pcm"] = task.inpdata

                rs.append(r)
                if len(rs) == task.userdata["total"]:
                    break
            return None

        th = threading.Thread(target=wait_proc, args=(self.__P, results))
        th.start()

        vad_segs = []
        head, tail = 0, len(pcm)
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(pcm[head : head + self.__sample_once_vad])
            if begin_ms >= 0:
                if begin_ms < end_ms:
                    vad_segs.append((begin_ms, end_ms))
            head += self.__sample_once_vad

        if head < tail:
            pcm_last = np.concatenate(
                [pcm[head:], np.zeros((self.__sample_once_vad - tail + head, ), dtype=np.float32)]
            )
            begin_ms, end_ms = self.__V.update_for_asr(pcm_last)
            if begin_ms >= 0 and begin_ms < end_ms:
                vad_segs.append((begin_ms, end_ms))
        
        begin_ms, end_ms = self.__V.update_for_asr(None)
        if begin_ms >= 0 and begin_ms < end_ms:
            vad_segs.append((begin_ms, end_ms))

        logger.debug(f"{self.__class__.__name__}: update_file: GOT {len(vad_segs)} segs, vad_segs={vad_segs}")
        
        for begin_ms, end_ms in vad_segs:
            begin_stamp = 16 * begin_ms; end_stamp = 16 * end_ms
            pcm_seg = pcm[begin_stamp:end_stamp]
            task = ATask(
                self.__mod_mask, 
                pcm_seg, 
                userdata={"begin_ms": begin_ms, "end_ms": end_ms, "total": len(vad_segs)}
            )
            self.__P.post_task(task)
            
        th.join()

        ## XXX: results 必须重新排序
        results.sort(key=lambda x: x["begin_ms"])

        if self.debug:
            for rs in results:
                txt = ''.join(rs['tokens'])
                logger.info(f"    GOT vad seg: {rs['begin_ms']}-{rs['end_ms']}, txt:{txt}, pcm:{rs['pcm']}")

        return results

    def __do_asr(self, begin_ms, end_ms, pcm=None) -> dict:
        if pcm is None:
            begin_samples = 16 * (begin_ms - self.__stamp_ms_cached_pcm)
            end_samples = 16 * (end_ms - self.__stamp_ms_cached_pcm)
            pcm = self.__cached_pcm[begin_samples:end_samples]
        task = ATask(self.__mod_mask, pcm, userdata={})
        self.__P.post_task(task)
        task.wait()
        rs = {
            "begin_ms": begin_ms,
            "end_ms": end_ms,
            "tokens": task.data["asr_dec_token"],
            "stamps": [(s[0] + begin_ms, s[1] + begin_ms) for s in task.data["asr_dec_stamp"]]
        }

        if self.debug:
            rs["pcm"] = task.inpdata
        return rs

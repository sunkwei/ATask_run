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
    def __init__(self, pipe=None, batch_size:int=1, debug=False):
        self.batch_size = batch_size
        self.debug = debug
        self.__cached_pcm = np.empty((0, ), np.float32)
        self.__stamp_ms_cached_pcm = 0
        self.__sample_once_vad = 6 * 16000
        self.thr_db_thresh = 0.003
        self.__cached_pcm_next_vad = 0  ## 保留下一段需要提交给 vad 的位置，相对 self.__cached_pcm 偏移
        self.__V = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        self.__mod_mask = mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP | mid.DO_SENSEVOICE
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
            logger.debug(f"{self.__class__.__name__}: update_stream: pcm.shape={pcm.shape}, last={last}, cached_pcm={self.__cached_pcm.shape[0]}, cached_begin_stamp:{self.__stamp_ms_cached_pcm}")

        self.__cached_pcm = np.concatenate([self.__cached_pcm, pcm])
        head, tail = self.__cached_pcm_next_vad, len(self.__cached_pcm)
        asr_result = []
        last_stamp = self.__stamp_ms_cached_pcm
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(self.__cached_pcm[head : head + self.__sample_once_vad])
            self.__cached_pcm_next_vad += self.__sample_once_vad    ## 下次提交 vad 的位置
            if begin_ms >= 0:
                rs = self.__do_asr(begin_ms, end_ms)
                asr_result.append(rs)
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

            begin_ms, end_ms = self.__V.update_for_asr(None)
            if begin_ms >= 0 and begin_ms < end_ms:
                rs = self.__do_asr(begin_ms, end_ms)
                asr_result.append(rs)

            ## 所有清空
            self.__V.reset()
            self.__cached_pcm = np.empty((0, ), np.float32)
            self.__stamp_ms_cached_pcm = 0
            self.__cached_pcm_next_vad = 0
        else:
            ## 删除不再需要的 pcm
            if last_stamp > self.__stamp_ms_cached_pcm:
                used_samples = 16 * (last_stamp - self.__stamp_ms_cached_pcm)
                self.__cached_pcm = self.__cached_pcm[used_samples:]
                self.__stamp_ms_cached_pcm = last_stamp
                ## 更新 vad 相对位置
                self.__cached_pcm_next_vad -= used_samples

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
        batch_size = self.batch_size
        results = []

        def wait_proc(pipe:APipe, rs:list):
            while 1:
                task = pipe.wait()
                if not task.userdata.get("batch", False):
                    r = {
                        "begin_ms": task.userdata["begin_ms"],
                        "end_ms": task.userdata["end_ms"],
                        "tokens": task.data["asr_dec_token"][0],
                        "stamps": task.data["asr_dec_stamp"][0],
                    }

                    if self.debug:
                        asr_predictor_infer = task.data["asr_predictor_infer"] #
                        pre_token_length = asr_predictor_infer[1][0]
                        r.update({
                            "asr pre_token_length": pre_token_length,
                        })

                    rs.append(r)
                else:
                    ## 批次模式：
                    for i in range(len(task.userdata["batch_seg_ms"])):
                        r = {
                            "begin_ms": task.userdata["batch_seg_ms"][i][0],
                            "end_ms": task.userdata["batch_seg_ms"][i][1],
                            "tokens": task.data["asr_dec_token"][i],
                            "stamps": task.data["asr_dec_stamp"][i],
                        }

                        if self.debug:
                            asr_predictor_infer = task.data["asr_predictor_infer"] #
                            pre_token_length = asr_predictor_infer[1][i]
                            r.update({
                                "asr pre_token_length": pre_token_length,
                            })

                        rs.append(r)

                if len(rs) == task.userdata["total"]:
                    break
            return None

        def cal_mean_audio(audio):
            if len(audio) > 100:
                audio_up = audio[audio>0]
                sort_d = np.sort(audio_up)
                keep = sort_d[-int(len(sort_d) * 0.1):]
                if len(keep) == 0:
                    return 0.0001
                return float(round(np.mean(keep),5)) + 0.0001
            else:
                return 0.0001
        
        th = threading.Thread(target=wait_proc, args=(self.__P, results))
        th.start()

        vad_segs = []
        head, tail = 0, len(pcm)
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(pcm[head : head + self.__sample_once_vad])
            if begin_ms >= 0:
                begin_stamp = 16 * begin_ms; end_stamp = 16 * end_ms
                pcm_seg = pcm[begin_stamp:end_stamp]

                if cal_mean_audio(pcm_seg) < self.thr_db_thresh:
                    continue

                vad_segs.append((begin_ms, end_ms))
            head += self.__sample_once_vad

        if head < tail:
            begin_ms, end_ms = self.__V.update_for_asr(pcm[head:], last=True)
            if begin_ms >= 0:
                begin_stamp = 16 * begin_ms; end_stamp = 16 * end_ms
                pcm_seg = pcm[begin_stamp:end_stamp]

                if cal_mean_audio(pcm_seg) >= self.thr_db_thresh:
                    vad_segs.append((begin_ms, end_ms))
        
        logger.debug(f"{self.__class__.__name__}: update_file: GOT {len(vad_segs)} segs, vad_segs={vad_segs}")

        if batch_size == 1:
            for i, (begin_ms, end_ms) in enumerate(vad_segs):
                begin_sample = 16 * begin_ms; end_sample = 16 * end_ms
                pcm_seg = pcm[begin_sample:end_sample]
                task = ATask(
                    self.__mod_mask,
                    pcm_seg,
                    userdata={"begin_ms": begin_ms, "end_ms": end_ms, "total": len(vad_segs)}
                )
                self.__P.post_task(task)
        else:
            ## XXX: 对 vad 片段根据长度进行排序，采用“桶”模式构造“批次数据”的 ATask
            ## 因为 asr vad 片段为 (0, 15秒)，所以可以构造 15 个桶，分别存储 (0, 1], (1, 2], ... (14, 15) 的片段
            buckets = [[] for _ in range(15)]   ## 每个桶存储 vad 片段序号
            for i, (begin_ms, end_ms) in enumerate(vad_segs):
                bucket_idx = (end_ms - begin_ms) // 1000
                buckets[bucket_idx].append(i)

            buckets = buckets[::-1]     ## 反序，防止慢慢增大
        
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
            fname = f"batch_{batch_size}_debug.txt"
            with open(fname, "w") as f:
                for r in results:
                    text = ''.join(r['tokens'])
                    f.write(f"{r['begin_ms']}\t{r['end_ms']}, pre_token_len:{r['asr pre_token_length']}, txt: {text}'\n")
            logger.warning(f"{fname} saved!!!!")

        if self.debug:
            for rs in results:
                txt = ''.join(rs['tokens'])
                logger.info(f"    GOT vad seg: {rs['begin_ms']}-{rs['end_ms']}, txt:{txt}, pcm:{rs['pcm']}")

        return results
    
    def __do_asr(self, begin_ms, end_ms, pcm=None) -> dict:
        logger.debug("__do_asr: begin_ms:{}, end_ms:{}, cache_samples from {} to {}".format(
            begin_ms, end_ms, self.__stamp_ms_cached_pcm, len(self.__cached_pcm) / 16 + self.__stamp_ms_cached_pcm
        ))
        if pcm is None:
            begin_samples = 16 * (begin_ms - self.__stamp_ms_cached_pcm)
            end_samples = 16 * (end_ms - self.__stamp_ms_cached_pcm)
            assert end_samples <= len(self.__cached_pcm)
            pcm = self.__cached_pcm[begin_samples:end_samples]
        task = ATask(self.__mod_mask, pcm, userdata={})
        self.__P.post_task(task)
        task.wait()
        rs = {
            "begin_ms": begin_ms,
            "end_ms": end_ms,
            "tokens": task.data["asr_dec_token"][0],
            "stamps": [(s[0] + begin_ms, s[1] + begin_ms) for s in task.data["asr_dec_stamp"][0]]
        }

        return rs

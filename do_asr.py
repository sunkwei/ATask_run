from logging.handlers import TimedRotatingFileHandler
import numpy as np
from typing import cast
import logging
import threading
from atask_run.models.asr_vad import Model_asr_vad
from atask_run.models.punc_bin import CT_Transformer
import atask_run.model_id as mid
from atask_run.apipe import APipe
from atask_run.atask import ATask
from atask_run.merge_asr import merge_asr
from atask_run.seg import seg_asr, alone_merge, find_non_overlapping_intervals, cal_mean_audio, cut_long_seg
import time
import os
from atask_run.del_str import opt_punc, get_coincident_area, count_elements_in_list, crop_winds
from atask_run.feat_cluster import sample_cluster
from atask_run.scentence_post import ScentencePost
curr_path = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger('asr_runner')

trfh = TimedRotatingFileHandler(filename=os.path.join(
    curr_path, "asr_runner.log"), interval=10, when="D", backupCount=1)
formatter = logging.Formatter(fmt="%(asctime)s %(message)s")
logger.addHandler(trfh)
trfh.setFormatter(formatter)
logger.setLevel(logging.DEBUG)


class ASRRunner:
    def __init__(self, pipe=None, batch_size: int = 1, debug=False):
        # ASR 模型不使用批次
        self.batch_size = 1  # batch_size
        self.debug = debug
        self.__cached_pcm = np.empty((0, ), np.float32)
        self.__stamp_ms_cached_pcm = 0
        self.__sample_once_vad = 6 * 16000
        self.thr_db_thresh = 0.003
        self.__cached_pcm_next_vad = 0  # 保留下一段需要提交给 vad 的位置，相对 self.__cached_pcm 偏移
        self.__V = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        self.__PUNC = CT_Transformer(model_dir="./model/punc")
        self.SP = ScentencePost()
        self.__mod_mask = mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE \
            | mid.DO_ASR_STAMP | mid.DO_SENSEVOICE | mid.DO_VOICEPRINT | mid.DO_4CLS | mid.DO_T5_ENCODER | mid.DO_T5_DEC1ST | mid.DO_T5_DECKVS

        if isinstance(pipe, APipe):
            self.__P = cast(APipe, pipe)
            self.__owner = False
        else:
            self.__P = APipe(model_mask=self.__mod_mask)
            self.__owner = True
        logger.info(
            f"ASRRunner: mod_mask={self.__mod_mask:b}, pipe={self.__P}")

    def __del__(self):
        del self.__V
        if self.__owner:
            del self.__P
        logger.info(f"ASRRunner: del\n")

    def update_stream(self, pcm: np.ndarray, last: bool = False) -> list:
        '''
            适合每次写入一段

            循环，每次 vad 得到一条片段，执行 asr 得到结果，再 vad ...
        '''
        if self.debug:
            logger.debug(
                f"{self.__class__.__name__}: update_stream: pcm.shape={pcm.shape}, last={last}, cached_pcm={self.__cached_pcm.shape[0]}, cached_begin_stamp:{self.__stamp_ms_cached_pcm}")

        self.__cached_pcm = np.concatenate([self.__cached_pcm, pcm])
        head, tail = self.__cached_pcm_next_vad, len(self.__cached_pcm)
        asr_result = []
        last_stamp = self.__stamp_ms_cached_pcm
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(
                self.__cached_pcm[head: head + self.__sample_once_vad])
            self.__cached_pcm_next_vad += self.__sample_once_vad  # 下次提交 vad 的位置
            if begin_ms >= 0:
                rs = self.__do_asr(begin_ms, end_ms)
                asr_result.append(rs)
                last_stamp = end_ms
            head += self.__sample_once_vad

        if last:
            if head < tail:
                pad_samples = self.__sample_once_vad - (tail - head)
                self.__cached_pcm = np.pad(
                    self.__cached_pcm, (0, pad_samples), mode='constant', constant_values=(0))
                begin_ms, end_ms = self.__V.update_for_asr(
                    self.__cached_pcm[head:])
                if begin_ms >= 0 and begin_ms < end_ms:
                    rs = self.__do_asr(begin_ms, end_ms)
                    asr_result.append(rs)

            begin_ms, end_ms = self.__V.update_for_asr(None)
            if begin_ms >= 0 and begin_ms < end_ms:
                rs = self.__do_asr(begin_ms, end_ms)
                asr_result.append(rs)

            # 所有清空
            self.__V.reset()
            self.__cached_pcm = np.empty((0, ), np.float32)
            self.__stamp_ms_cached_pcm = 0
            self.__cached_pcm_next_vad = 0
        else:
            # 删除不再需要的 pcm
            if last_stamp > self.__stamp_ms_cached_pcm:
                used_samples = 16 * (last_stamp - self.__stamp_ms_cached_pcm)
                self.__cached_pcm = self.__cached_pcm[used_samples:]
                self.__stamp_ms_cached_pcm = last_stamp
                # 更新 vad 相对位置
                self.__cached_pcm_next_vad -= used_samples

        return asr_result

    def update_file(self, pcm: np.ndarray) -> list:
        '''
            适合一次 asr 处理一节课
            先做 Vad 得到所有片段，然后扔到 APipe 中，启动一个工作线程，接受结果
        '''
        logger.info(
            f"{self.__class__.__name__}: update_file: pcm.shape={pcm.shape}, "
            f"duration:{len(pcm)/16000:.03f} seconds"
        )
        batch_size = self.batch_size
        results_para = []
        results_sensevoice = []

        def wait_proc(pipe: APipe, rs: list, rs_sensevoice: list):
            while 1:
                task = pipe.wait()
                if 1:
                    r = {
                        "timestamp": task.data["asr_dec_stamp"][0],
                        "raw_tokens": task.data["asr_dec_raw_tokens"][0],
                        "preds": task.data["asr_dec_preds"][0],
                        "idx": task.userdata["idx"],
                        "begin_ms": task.userdata["begin_ms"],
                        "asr_type": task.data["asr_type"][0],
                        "audio_db": task.userdata["audio_db"],
                    }

                    rs.append(r)

                    r1 = {
                        "timestamp": task.data["asr_sensevoice_result"][0]["timestamp"],
                        "raw_tokens": task.data["asr_sensevoice_result"][0]["raw_tokens"],
                        "preds": task.data["asr_sensevoice_result"][0]["preds"],
                    }

                    rs_sensevoice.append(r1)

                if len(rs) == task.userdata["total"]:
                    break
            return None

        th = threading.Thread(target=wait_proc, args=(
            self.__P, results_para, results_sensevoice))
        th.start()

        begin_time = time.time()
        ########### VAD begin!###########
        vad_segs = []
        audio_db_list = []
        head, tail = 0, len(pcm)
        while head + self.__sample_once_vad <= tail:
            begin_ms, end_ms = self.__V.update_for_asr(
                pcm[head: head + self.__sample_once_vad])
            if begin_ms >= 0:
                begin_stamp = 16 * begin_ms
                end_stamp = 16 * end_ms
                pcm_seg = pcm[begin_stamp:end_stamp]

                audio_db = cal_mean_audio(pcm_seg)
                if audio_db < self.thr_db_thresh:
                    continue

                vad_segs.append((begin_ms, end_ms))
                audio_db_list.append(audio_db)

            head += self.__sample_once_vad

        if head < tail:
            begin_ms, end_ms = self.__V.update_for_asr(pcm[head:], last=True)
            if begin_ms >= 0:
                begin_stamp = 16 * begin_ms
                end_stamp = 16 * end_ms
                pcm_seg = pcm[begin_stamp:end_stamp]

                audio_db = cal_mean_audio(pcm_seg)
                if audio_db >= self.thr_db_thresh:
                    vad_segs.append((begin_ms, end_ms))
                    audio_db_list.append(audio_db)

        vad_time = time.time()
        ########### VAD end! ###########
        logger.debug(
            f"{self.__class__.__name__}: update_file: GOT {len(vad_segs)} segs")
        logger.debug(
            f"update_file: vad time cost: {vad_time-begin_time:.03f} seconds")

        ########### ASR begin! ###########
        if batch_size == 1:
            for i, (begin_ms, end_ms) in enumerate(vad_segs):
                begin_sample = 16 * begin_ms
                end_sample = 16 * end_ms
                pcm_seg = pcm[begin_sample:end_sample]
                task = ATask(
                    mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP | mid.DO_SENSEVOICE,
                    pcm_seg,
                    userdata={"begin_ms": begin_ms, "end_ms": end_ms, "total": len(
                        vad_segs), "idx": i, "audio_db": audio_db_list[i]}
                )
                self.__P.post_task(task)
        else:
            # XXX: 对 vad 片段根据长度进行排序，采用“桶”模式构造“批次数据”的 ATask
            # 因为 asr vad 片段为 (0, 15秒)，所以可以构造 15 个桶，分别存储 (0, 1], (1, 2], ... (14, 15) 的片段
            buckets = [[] for _ in range(15)]  # 每个桶存储 vad 片段序号
            for i, (begin_ms, end_ms) in enumerate(vad_segs):
                bucket_idx = (end_ms - begin_ms) // 1000
                buckets[bucket_idx].append(i)

            buckets = buckets[::-1]  # 反序，防止慢慢增大

        th.join()
        ########### ASR end! ###########
        asr_time = time.time()
        logger.debug(
            f"update_file: asr time cost: {asr_time-vad_time:.03f} seconds")

        # 双asr文本内容合并策略
        time_list_all, txt_list, crop_punc = merge_asr(
            results_para, results_sensevoice)

        # 文本切割小片段
        TT_single, group_dex, crop_time_list = seg_asr(
            pcm, txt_list, time_list_all)

        seg_asr_time = time.time()
        logger.debug(
            f"seg_asr time cost: {seg_asr_time-asr_time:.03f} seconds")

        ########## 声纹############
        FF = []
        sort_dex = []

        def wait_sv(pipe: APipe, F: list, index: list):
            while 1:
                task = pipe.wait()
                sv_feats = task.data["sv_feat"][0]
                F.append(sv_feats)
                index.append(task.userdata["dex"])

                if len(F) == task.userdata["total"]:
                    break

        th = threading.Thread(target=wait_sv, args=(self.__P, FF, sort_dex))
        th.start()

        for i in range(len(TT_single)):
            s = TT_single[i][0]
            e = TT_single[i][1]
            part_wav = pcm[int(s*16000):int(e*16000)]
            task = ATask(
                mid.DO_VOICEPRINT,
                part_wav,
                userdata={"dex": i, "total": len(TT_single)}
            )
            self.__P.post_task(task)

        ########## 等待声纹结果时，进行标点预测 ############
        if len(txt_list) == 0:
            punc = []
        else:
            result_punc = self.__PUNC(" ".join(txt_list))
            punc = result_punc[1]
            if len(punc) >= 3:
                punc = opt_punc(txt_list, punc, crop_time_list)

            if len(crop_punc) > 0:
                punc = np.array(punc)
                for c in crop_punc:
                    punc[c[0] + 1:c[1]] = 1
                    if punc[c[0]] == 1:
                        punc[c[0]] = 2
                    if punc[c[1]] == 1:
                        punc[c[1]] = 2

        punc_time = time.time()
        logger.debug(f"punc time cost: {punc_time-seg_asr_time:.03f} seconds")

        th.join()
        sv_time = time.time()
        logger.debug(
            f"sv & punc time cost: {sv_time-seg_asr_time:.03f} seconds")

        ########### 声纹、标点预测结束 ############
        FF_single = np.array(FF)
        sort_dex = np.argsort(sort_dex)
        FF_single = FF_single[sort_dex]

        # 处理单字，前后合并
        rm_dex_alone = alone_merge(TT_single, FF_single, group_dex)
        # 从数据里删除被合并的单字的数据
        TT_single = np.delete(TT_single, rm_dex_alone, axis=0)
        FF_single = np.delete(FF_single, rm_dex_alone, axis=0)

        main_interval = np.array([0, len(pcm) / 16000])
        no_single_time = find_non_overlapping_intervals(
            main_interval, TT_single)

        ########### 对人声片段的声纹特征聚类,确定老师类 ############
        if len(FF_single) > 0:
            LL_O = sample_cluster(FF_single, TT_single, no_single_time)
            LL_O = LL_O.astype(int)
        else:
            LL_O = []

        ########## 预测四分类 ############
        v4_results = []
        v4_td = []
        v4_time = []

        def wait_v4(pipe: APipe, r: list, td: list, t: list):
            while 1:
                task = pipe.wait()
                r.append(task.data["v4_infer"])
                td.append(task.data["td"])
                t.append([task.userdata["begin_time"],
                         task.userdata["end_time"]])

                if len(r) == task.userdata["total"]:
                    break
            return None

        th = threading.Thread(target=wait_v4, args=(
            self.__P, v4_results, v4_td, v4_time))
        th.start()

        no_single_time = cut_long_seg(no_single_time)
        for i in range(len(no_single_time)):
            s = no_single_time[i][0]
            e = no_single_time[i][1]
            part_wav = pcm[int(s*16000):int(e*16000)]
            task = ATask(
                mid.DO_4CLS,
                part_wav,
                userdata={"begin_time": s, "end_time": e,
                          "total": len(no_single_time)}
            )
            self.__P.post_task(task)

        role_data = np.array([])
        scentence_data = {'data': [], 'statistics': {}}
        if len(LL_O) > 0:
            role_data = np.hstack((TT_single, np.array([LL_O]).T))
            ############ 每个字标注角色 ############
            txt_label = []
            last_label = -1
            for i in range(len(crop_time_list)):
                time_t = np.array([crop_time_list[i][0], crop_time_list[i][1]])
                area_list = get_coincident_area(time_t, role_data)
                curr_label = int(role_data[np.argmax(area_list), -1])

                if txt_list[i] in self.SP.first_rm and curr_label != last_label:
                    # 结束的语气词不能是一个新角色的开始
                    # 一句话的结束字不能是一个新角色的开始
                    curr_label = last_label

                txt_label.append(curr_label)

                last_label = curr_label

            ############# 统计口头禅 ############
            mantra_data = count_elements_in_list(
                txt_list, self.SP.mantra)  # 统计口头禅

            ############# 对于停顿大的，尝试增加断句 ############
            crop_thresh = 1
            split_dex = crop_winds(
                crop_time_list, txt_list, crop_thresh=crop_thresh)

            # ############ 进行句子后处理（断句、情绪、itn）############
            scentence_data = self.SP(
                txt_list, crop_time_list, punc, split_dex, txt_label, mantra_data)

        th.join()
        v4_scentence_time = time.time()
        logger.debug(
            f"v4 & cluster & scentence_post time cost: {v4_scentence_time-sv_time:.03f} seconds")

        ########## 对纯英文句子英译汉 #########
        tr_num = 0
        for dex, data in enumerate(scentence_data["data"]):
            if len(data["translation"]) > 0:
                # 如果是纯英文句子，则进行翻译
                task = ATask(
                    mid.DO_T5_ENCODER | mid.DO_T5_DEC1ST | mid.DO_T5_DECKVS,
                    data["translation"],
                    userdata={"dex": dex}
                )
                self.__P.post_task(task)
                tr_num += 1

        trans_results = {}

        def wait_trans(pipe: APipe, tr: dict, num: int):
            while 1:
                task = pipe.wait()
                tr[task.userdata["dex"]] = task.data["trans_post"]

                if len(tr) == num:
                    break
            return None

        if tr_num > 0:
            th = threading.Thread(target=wait_trans, args=(
                self.__P, trans_results, tr_num))
            th.start()

            th.join()
            for k in trans_results:
                scentence_data["data"][k]["translation"] = trans_results[k]

            t5_time = time.time()
            logger.debug(
                f"t5 time cost: {t5_time-v4_scentence_time:.03f} seconds")

        if len(role_data) > 0:
            single_data = {"role": role_data[:, -1].astype(int), "time": role_data[:, :2], "v4": [
                "single"] * len(role_data), "txt": [], "txt_time": []}
        else:
            single_data = {"role": np.array([], dtype=np.int32), "time": np.array(
                [], dtype=np.float32), "v4": [], "txt": [], "txt_time": []}

        # 排序
        v4_time = np.array(v4_time)
        sort_dex = np.argsort(v4_time[:, 0], axis=0)

        # 生成每秒的no_single四分类数据
        no_single_v4_sec = []
        no_single_time_sec = []
        for i in sort_dex:
            s = v4_time[i][0]
            t = v4_time[i][1]

            r, td = v4_results[i], v4_td[i]
            start = s
            for ii in range(len(r)):
                no_single_v4_sec.append(r[ii])
                no_single_time_sec.append([start, start + round(td[ii], 2)])

                start = start + round(td[ii], 2)

        no_single_data = {"time": no_single_time, "time_td": v4_td, "v4": v4_results,
                          "time_sec": np.array(no_single_time_sec), "v4_sec": no_single_v4_sec}

        logger.debug(f"all time cost: {time.time()-begin_time:.03f} seconds")

        return single_data, no_single_data, scentence_data

    def __do_asr(self, begin_ms, end_ms, pcm=None) -> dict:
        # 记录调试信息：开始时间、结束时间和缓存音频范围
        logger.debug("__do_asr: begin_ms:{}, end_ms:{}, cache_samples from {} to {}".format(
            begin_ms, end_ms, self.__stamp_ms_cached_pcm, len(
                self.__cached_pcm) / 16 + self.__stamp_ms_cached_pcm
        ))
        # 如果未提供音频数据，则从缓存中提取指定时间段的音频
        if pcm is None:
            begin_samples = 16 * \
                (begin_ms - self.__stamp_ms_cached_pcm)  # 转换为样本数
            end_samples = 16 * \
                (end_ms - self.__stamp_ms_cached_pcm)      # 转换为样本数
            assert end_samples <= len(self.__cached_pcm)  # 确保请求的样本数不超过缓存长度
            pcm = self.__cached_pcm[begin_samples:end_samples]  # 从缓存中提取音频段
        # 创建ASR任务并提交到管道
        task = ATask(self.__mod_mask, pcm, userdata={})
        self.__P.post_task(task)  # 提交任务到处理管道
        task.wait()  # 等待任务完成
        # 构建并返回结果字典，包含时间戳、识别文本和对应的时间戳信息
        rs = {
            "begin_ms": begin_ms,
            "end_ms": end_ms,
            "tokens": task.data["asr_dec_token"][0],  # 识别的文本token
            # 相对时间戳转换为绝对时间戳
            "stamps": [(s[0] + begin_ms, s[1] + begin_ms) for s in task.data["asr_dec_stamp"][0]]
        }

        return rs

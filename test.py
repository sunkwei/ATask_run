import logging
from src.apipe import APipe, APipeWrap
from src.atask import ATask, ATask_Quit
import src.model_id as mid
from typing import List, cast
import numpy as np
import threading
import unittest
from src.timeused import TimeUsed, TimeUsedSum
from src.models.asr_vad import Model_asr_vad
from do_asr import ASRRunner
import soundfile

# 如果 windows 平台，需要使用不同的路径
import os.path as osp
import sys

if sys.platform == 'win32':
    TEST_FNAME0 = "p:/tmp/test_asr/726_part1.wav"
    TEST_726 = "n:/videos/726/teacher.wav"
    RESULT_PATH = "p:/tmp"
else:
    TEST_FNAME0 = "/media/pub/tmp/test_asr/726_part1.wav"
    TEST_726 = "/media/pub/videos/726/teacher.wav"
    RESULT_PATH = "/media/pub/tmp"

logger = logging.getLogger("test")

class Test(unittest.TestCase):
    def test_vad_1s(self):
        M = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        pcm, sr = soundfile.read(TEST_FNAME0)
        pcm = pcm.astype(np.float32)
        head, tail = 0, len(pcm)
        step = 16000    # 1秒一段的调用
        segs = []
        while head < tail:
            N = min(step, tail - head)
            segs.extend(M.update(pcm[head : head + N]))
            head += N
        segs.extend(M.update(None))

        ## 生成 audacity label 文件
        with open(osp.join(RESULT_PATH, f"test_vad_raw_{step}.txt"), "w") as f:
            for seg in segs:
                f.write(f"{seg[0]/1000.0:.03f}\t{seg[1]/1000.0:.03f}\tv\n")

        self.assertEqual(len(segs), 13)
        self.assertEqual(segs[0], [0, 1200])
        self.assertEqual(segs[1],[2750, 4090])
        self.assertEqual(segs[2], [4990, 6080])
        self.assertEqual(segs[-1], [29430, 31530])

    def test_vad_6s(self):
        M = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        pcm, sr = soundfile.read(TEST_FNAME0)
        pcm = pcm.astype(np.float32)
        head, tail = 0, len(pcm)
        step = 96000    # 6秒一段的调用
        segs = []
        while head < tail:
            N = min(step, tail - head)
            segs.extend(M.update(pcm[head : head + N]))
            head += N
        segs.extend(M.update(None))

        ## 生成 audacity label 文件
        with open(osp.join(RESULT_PATH, f"test_vad_raw_{step}.txt"), "w") as f:
            for seg in segs:
                f.write(f"{seg[0]/1000.0:.03f}\t{seg[1]/1000.0:.03f}\tv\n")

        self.assertEqual(len(segs), 13)
        self.assertEqual(segs[0], [0, 1200])
        self.assertEqual(segs[1],[2780, 4150])
        self.assertEqual(segs[2], [5090, 6180])
        self.assertEqual(segs[-1], [29950, 32070])

    def test_vad_for_asr(self):
        M = Model_asr_vad("./model/asr_vad/asr_vad.onnx")
        pcm, sr = soundfile.read(TEST_FNAME0)
        pcm = pcm.astype(np.float32)
        head, tail = 0, len(pcm)
        step = 96000    # 6秒一段的调用
        segs = []
        while head < tail:
            N = min(step, tail - head)
            begin_ms, end_ms = M.update_for_asr(pcm[head : head + N])
            if begin_ms >= 0:
                segs.append([begin_ms, end_ms])
            head += N
        begin_ms, end_ms = M.update_for_asr(None)
        if begin_ms >= 0:
            segs.append([begin_ms, end_ms])

        with open(osp.join(RESULT_PATH, f"test_vad_for_asr.txt"), "w") as f:
            for seg in segs:
                f.write(f"{seg[0]/1000.0:.03f}\t{seg[1]/1000.0:.03f}\tv\n")

        self.assertEqual(len(segs), 4)
        self.assertEqual(segs[0], [0, 1200])
        self.assertEqual(segs[1], [2780, 15740])
        self.assertEqual(segs[2], [17480, 20190])
        self.assertEqual(segs[3], [21510, 32070])

    def test_726_asr_stream(self):
        '''
        模拟流式 asr：
            loop:
                读1分钟pcm
                做 vad 得到 asr 片段
                对所有 asr 片段执行 asr
        
        :param self: Description
        '''
        wav_fname = TEST_726
        with APipeWrap(model_mask=mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP) as pipe:
            sess = ASRRunner(pipe)
            pcm, sr = soundfile.read(wav_fname, dtype="float32", frames=-1)
            with TimeUsed(f"asr_update_stream duration:{len(pcm)/16000:.03f} seconds"):
                asr_result = sess.update_stream(pcm, last=True)

        ## 存储为 audacity 标签文件
        with open(osp.join(RESULT_PATH, "test_asr_726_stream.txt"), "w") as f:
            for r in asr_result:
                begin_ms = r["begin_ms"]
                end_ms = r["end_ms"]
                tokens = r["tokens"]
                stamps = r["stamps"]

                txt = ''.join(tokens)
                f.write(f"{begin_ms/1000:.03f}\t{end_ms/1000:.03f}\t{txt}\n")

    def test_726_asr_file(self):
        '''
            模拟文件模式

        '''
        wav_fname = TEST_726
        with APipeWrap(model_mask=mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP) as pipe:
            sess = ASRRunner(pipe)
            pcm, sr = soundfile.read(wav_fname, dtype="float32", frames=-1)
            with TimeUsed(f"asr_update_file duration:{len(pcm)/16000:.03f} seconds"):
                asr_result = sess.update_file(pcm)

        ## 存储为 audacity 标签文件
        with open(osp.join(RESULT_PATH, "test_asr_726_file.txt"), "w") as f:
            for r in asr_result:
                begin_ms = r["begin_ms"]
                end_ms = r["end_ms"]
                tokens = r["tokens"]
                stamps = r["stamps"]

                txt = ''.join(tokens)
                f.write(f"{begin_ms/1000:.03f}\t{end_ms/1000:.03f}\t{txt}\n")


if __name__ == "__main__":
    import sys, json
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument("--test_all", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--build_model_config", action="store_true")
    ap.add_argument("--wav", type=str, default="")
    args = ap.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            filemode="w",
            filename="./test.log",
            format="%(asctime)s %(levelname)-8s %(message)s",
        )

    if args.build_model_config:
        from src.model_desc import build_default_model_configs, load_model_config
        build_default_model_configs()
        cfgs = load_model_config(config_path="./config_temp")
        print(json.dumps(cfgs, indent=2, ensure_ascii=False))
        for cfg in cfgs:
            print(f"mod: {cfg['model_path']}")
        sys.exit(0)

    elif args.test_all:
        unittest.main(argv=['first-arg-is-ignored'])
        tus = TimeUsedSum()
        tus.dump()
        sys.exit(0)
    

    def test():
        def load_test_pcms(path) -> List[np.ndarray]:
            import soundfile, os
            pcms = []
            for fname in os.listdir(path):
                if os.path.splitext(fname)[1] == ".wav":
                    pcm, sr = soundfile.read(os.path.join(path, fname))
                    pcms.append(pcm)
            return pcms
        
        ## 模拟测试多段 pcm
        pcms = load_test_pcms("/media/pub/tmp/test_asr")

        ## 创建 Pipe
        mod_mask = mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP
        pipe3 = APipe(model_mask=mod_mask)
        pipe1 = APipe(model_mask=mod_mask, debug_one_thread=True)

        ## 启动线程，读取结果
        def wait(pipe):
            count = len(pcms)
            while count > 0:
                task = pipe.wait()
                count -= 1
                token = task.data["asr_dec_token"]
                text = ''.join(token)
                print(f"#{count}, {text}")
            logger.info(f"wait proc done, count: {len(pcms)}")
            
        th = threading.Thread(target=wait, args=(pipe3,))
        th.start()

        with TimeUsed("pipe 3threads"):
            for pcm in pcms:
                task = ATask(
                    mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP,
                    pcm,
                    userdata={}
                )
                pipe3.post_task(task)

            th.join()

        th = threading.Thread(target=wait, args=(pipe1,))
        th.start()

        with TimeUsed("pipe 1thread"):
            for pcm in pcms:
                task = ATask(
                    mid.DO_ASR_ENCODE | mid.DO_ASR_PREDICTOR | mid.DO_ASR_DECODE | mid.DO_ASR_STAMP,
                    pcm,
                    userdata={}
                )
                pipe1.post_task(task)

            th.join()

        tus = TimeUsedSum()
        tus.dump()
        logger.info("all done")
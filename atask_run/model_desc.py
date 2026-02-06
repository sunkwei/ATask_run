'''
    描述模型，使用 yaml 配置
    每个模型有唯一的 mid （使用 DO_XXX ... ）
    模型的路径，后端，输入形状等
    不同的配置下，后端还有自己的配置，如 device_id 等

    模型存储在 ./model 目录下，配置存储在 ./config 目录下
'''

import yaml
from . import model_id as mid
from typing import List, Dict
import logging

logger = logging.getLogger("model_desc")

def load_model_config(
    config_path="./config",
    mod_mask:int=-1,
) -> List[Dict]:
    '''
    根据 mod_mask 从 config_path 目录中加载所有 yaml 文件
    并根据依赖关系返回模型列表

    如果依赖关系因为 mod_mask 不满足，抛出异常
    '''
    from pathlib import Path
    P = Path(config_path)
    cfgs = []
    for f in P.glob("*.yaml"):
        with open(f, encoding="utf8") as f:
         cfg = yaml.load(f, Loader=yaml.loader.FullLoader)
        if "mid" not in cfg:
            ## 非有效模型配置
            continue

        if (mod_mask & cfg["mid"]) == 0:
            logging.warning(f"skiping {f} for mod mask:{mod_mask:b}")
            continue

        cfgs.append(cfg)

    ## 检查依赖关系
    for i, cfg in enumerate(cfgs):
        deps = cfg["depend"]
        for dep in deps:
            if dep not in [c["mid"] for c in cfgs]:
                raise Exception(f"{cfg['model_path']} depend: {dep} is not found")

    ## 根据依赖关系排序 cfgs，被依赖的模型排在前面
    for i, _ in enumerate(cfgs):
        for j, cfg in enumerate(cfgs):
            if cfg["mid"] == cfgs[i]["mid"] or cfg["mid"] in cfgs[i]["depend"]:
                cfgs[j], cfgs[i] = cfgs[i], cfgs[j]

    return cfgs[::-1]

def build_default_model_configs(
    config_path="./config_temp", 
    model_path="./model"
):
    ''' 为已知的模型生成默认的配置文件 ...
        默认 onnxruntime 使用 cpu 推理

        相当于生产模板文件，然后根据需要修改
    '''
    import os
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    ## 生成 APipe.yaml
    with open(os.path.join(config_path, "APipe.yaml"), "w") as f:
        cfg = {
            "Q_inp_size": 4,
            "Q_inp_sub_size": -1,
            "Q_pre_size": 16,
            "Q_infer_ize": -1,
            "Q_result_size": -1,

            "E_pre_num_thread": 2,
            "E_infer_num_thread": 2,
            "E_post_num_thread": 2,
        }
        yaml.dump(cfg, f)

    def save(name, mid, inps, dep=[]):
        desc = {
            "mid": mid,         ## DO_ACT, DO_FACEDET, ...
            "name": name,       ## XXX: 类实现名字总是 f"Model_{name}"，并且在 src.models.{name}.py 中实现
            "model_path": f"{model_path}/{name}.onnx",      ## 模型路径，如果 backend 中提供，将覆盖该名字
            "input": inps,
            "depend": dep,
            "backend": [
                {
                    "type": "onnxruntime",
                    "model_path": f"{model_path}/{name}.onnx",  ## backend 中的 model_path 将覆盖上面的 model_path
                    "backend_cfg": {
                        "force_cpu": True,          ## 是否强制使用 cpu
                        "balance_ratio": 1,             ## 用于动态复制均衡比例，如果使用加速卡，设置 0.1 则 90% 分配给 gpu，10% 分配给 cpu

                        "intra_op_num_threads": 1,  ## onnx 算子线程数
                        "inter_op_num_threads": 1,

                        "providers": [                  ## onnx providers, 与 providers_cfg 的长度必须相同
                            "TensorrtExecutionProvider",
                            "CUDAExecutionProvider",
                            "CANNExecutionProvider",
                            "CPUExecutionProvider",
                        ],

                        "providers_cfg": [  ## 注意，比如与 providers 数量相同
                            {
                                ## TensorrtExecutionProvider
                                "device_id": 0,
                                "trt_engine_cache_enable": True,
                                "trt_engine_cache_path": "./run/cache",
                                "trt_engine_cache_prefix": f"trt_{name}",
                            },
                            {
                                ## CUDAExecutionProvider
                                "device_id": 0,
                            },
                            {
                                ## CANNExecutionProvider
                                "device_id": 0,
                                "enable_cann_graph": True,
                            },
                            {
                                ## CPUExecutionProvider
                            },
                        ]
                    },
                },
                ## XXX: 此处支持多个 backend 实现
            ],
        }
        with open(f"./{config_path}/{name}.yaml", 'w') as f:
            yaml.dump(desc, f)
    
    save("act", mid.DO_ACT, [{"name":"input","dtype":"float32","shape":[1,3,544,960]}], [])
    save("facedet", mid.DO_FACEDET, [{"name":"input","dtype":"float32","shape":[1,3,544,960]}], [])
    save("face_score", mid.DO_FACE_SCORE, [{"name":"input","dtype":"float32","shape":[1,3,112,112]}], [mid.DO_FACEDET])
    save("facerec", mid.DO_FACEREC, [{"name":"input","dtype":"float32","shape":[1,3,112,112]}], [mid.DO_FACEDET, mid.DO_FACE_SCORE])
    save("faceori", mid.DO_FACEORI, [{"name":"input","dtype":"float32","shape":[1,3,192,192]}], [mid.DO_FACEDET, mid.DO_FACE_SCORE])
    save("raisehandcls", mid.DO_RAISEHANDCLS, [{"name":"input","dtype":"float32","shape":[1,3,224,224]}], [mid.DO_ACT])
    save("findteacher", mid.DO_FINDTEACHER, [{"name":"input","dtype":"float32","shape":[1,3,544,960]}], [])
    save("classtype", mid.DO_CLASSTYPE, [{"name":"input","dtype":"float32","shape":[1,3,640,360]}], [])

    save(
        "asr_vad",
        mid.DO_ASR_VAD,
        [
            {"name":"speech","dtype":"float32","shape":[1,30,400],},
            {"name":"in_cache0","dtype":"float","shape":[1,128,19,1],},
            {"name":"in_cache1","dtype":"float","shape":[1,128,19,1],},
            {"name":"in_cache2","dtype":"float","shape":[1,128,19,1],},
            {"name":"in_cache3","dtype":"float","shape":[1,128,19,1],},
        ],
        [],
    )
    save(
        "asr_enc", 
        mid.DO_ASR_ENCODE, 
        [
            {"name":"speech","dtype":"float32","shape":[1, -1, 560]},
            {"name":"mask3","dtype":"float32","shape":[1,-1,1]},
            {"name":"mask4","dtype":"float32","shape":[1,1,1,-1]},
        ], 
        [],
    )
    save(
        "asr_predictor",
        mid.DO_ASR_PREDICTOR,
        [
            {"name":"enc","dtype":"float32","shape":[1,-1,512]},
            {"name":"enc_mask","dtype":"float32","shape":[1,1,-1]},
        ],
        [mid.DO_ASR_ENCODE],
    )
    save(
        "asr_dec",
        mid.DO_ASR_DECODE,
        [
            {"name":"enc","dtype":"float","shape":[1,-1,512]},
            {"name":"enc_mask","dtype":"float","shape":[1,1,-1]},
            {"name":"pre_acoustic_embeds","dtype":"float","shape":[1,-1,512]},
            {"name":"pre_token_mask","dtype":"float32","shape":[1,1,-1]},
        ],
        [mid.DO_ASR_ENCODE, mid.DO_ASR_PREDICTOR],
    )
    save(
        "asr_stamp",
        mid.DO_ASR_STAMP,
        [
            {"name":"enc","dtype":"float","shape":[1,-1,512]},
            {"name":"mask","dtype":"float","shape":[1,1,-1]},
            {"name":"pre_token_length","dtype":"int32","shape":[1]},
        ],
        [mid.DO_ASR_ENCODE, mid.DO_ASR_PREDICTOR, mid.DO_ASR_DECODE],
    )

    save(
        "asr_sensevoice",
        mid.DO_SENSEVOICE,
        [
            {"name":"speech","dtype":"float","shape":[1,-1,560]},
            {"name":"speech_lengths","dtype":"int32","shape":[1]},
            {"name":"language","dtype":"int32","shape":[1]},
            {"name":"textnorm","dtype":"int32","shape":[1]},
        ],
        [mid.DO_ASR_ENCODE, mid.DO_ASR_PREDICTOR, mid.DO_ASR_DECODE, mid.DO_ASR_STAMP],
    )
    return None

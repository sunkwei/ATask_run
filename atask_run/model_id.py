DO_ACT = 1
DO_FACEDET = 2
DO_FACEREC = 4
DO_FACEORI = 8
DO_EXPRESS = 16
DO_4CLS = 32
DO_VOICEPRINT = 64
DO_RAISEHANDCLS = 128
DO_FINDTEACHER = 256
DO_CLASSTYPE = 512
# DO_FACEFLT = 1024
DO_FACE_SCORE = 2048
DO_WBCLS = 4096

DO_ASR_ENCODE = 0x10000
DO_ASR_VAD =    0x20000
DO_T5 =         0x40000
DO_PUNC =       0x80000
DO_ASR_DECODE = 0x100000

## 声音5分类，将代替 DO_4CLS
DO_SOUND_C5 = 0x200000

## ocr
DO_OCR_DET = 0x400000
DO_OCR_REC = 0x800000

## pose
DO_POSE = 0x1000000

## 黑板，大屏识别
DO_BOARD_DET = 0x20000000
DO_PHONE_DET = 0x40000000

DO_ASR_PREDICTOR = 0x80000000
DO_ASR_STAMP = 0x100000000
DO_SENSEVOICE = 0x200000000
DO_T5_ENCODER = 0x400000000
DO_T5_DEC1ST = 0x800000000
DO_T5_DECKVS = 0x1000000000

def todo2str(todo:int) -> str:
    if todo == 0:
        return "None"
    
    s = []
    for k, v in globals().items():
        if isinstance(v, int) and todo & v == v:
            s.append(k)
    return "|".join(s)
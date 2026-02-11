from unittest import TestCase
import unittest
from atask_run.atask import ATask
from atask_run.apipe import APipeWrap
import atask_run.model_id as mid
from atask_run.timeused import TimeUsed
import cv2

class TestOcr(TestCase):
    def test_ocr_det(self):
        img = cv2.imread("picture/ocr3.jpg")
        if img.size == 0:
            raise FileNotFoundError("picture/ocr.jpg not found")
        
        with APipeWrap(mid.DO_OCR_DET | mid.DO_OCR_REC, profile=True) as pipe:
            with TimeUsed("ocr_det"):
                task = ATask(mid.DO_OCR_DET | mid.DO_OCR_REC, img, userdata={})
                pipe.post_task(task)
                task = task.wait()
            boxes = task.data["ocr_det_result"][0]  # 一张图象
            for i,box in enumerate(boxes):
                # box: (N, 2), N 个点的坐标，在原图绘制多边形
                cv2.polylines(img, [box], True, (0, 255, 0), thickness=2)
                txt = task.data["ocr_rec_result"][0][i][0]
                print(f"{i}: {box}, text:'{txt}'")

            cv2.imshow("ocr_det", img)
            cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
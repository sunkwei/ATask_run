''' 测试用例，测试图像任务
'''

from unittest import TestCase
import unittest
from atask_run.apipe import ATask, APipe, APipeWrap
import atask_run.model_id as mid
import cv2
from atask_run.timeused import TimeUsed, TimeUsedSum

class ImageTestCase(TestCase):

    def _test_action_B1(self):
        fnames = [
            "picture/teacher.jpg",
            "picture/student.jpg",
        ]

        images = [ cv2.imread(fname) for fname in fnames ]
        
        with APipeWrap(mid.DO_ACT) as pipe:
            result1 = []; result2 = []

            for i in range(1):
                task1 = ATask(todo=mid.DO_ACT, inpdata=tuple(images), userdata={})
                with TimeUsed("test_action_B1"):            
                    pipe.post_task(task1)
                    result1 = pipe.wait().data["act_result"]

                task2 = ATask(todo=mid.DO_ACT, inpdata=tuple(images), userdata={})
                with TimeUsed("test_action_B1"):            
                    pipe.post_task(task2)
                    result2 = pipe.wait().data["act_result"]

            result = [ result1[0], result2[0] ]
            
            for i, r in enumerate(result):
                img0 = images[i]
                for j in range(len(r)):
                    x1, y1, x2, y2, score, cid = r[j]
                    cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # cv2.imshow("image", img0)
                # cv2.waitKey(0)

    def _test_action_B8(self):
        fnames = [
            "picture/teacher.jpg",
            "picture/student.jpg",
        ]

        images = [ cv2.imread(fname) for fname in fnames ]
        images.extend(images)
        images.extend(images)
        images.extend(images)
        
        with APipeWrap(mid.DO_ACT) as pipe:
            result = []
            for i in range(1):
                task = ATask(todo=mid.DO_ACT, inpdata=tuple(images), userdata={})
        
                with TimeUsed("test_action_B8"):
                    pipe.post_task(task)
                    result = pipe.wait().data["act_result"]
            
            for i, r in enumerate(result):
                img0 = images[i]
                for j in range(len(r)):
                    x1, y1, x2, y2, score, cid = r[j]
                    cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # cv2.imshow("image", img0)
                # cv2.waitKey(0)

    def test_face_B1(self):
        fnames = [
            "picture/teacher.jpg",
            "picture/student.jpg",
        ]

        images = [ cv2.imread(fname) for fname in fnames ]
        
        todo0 = mid.DO_ACT | mid.DO_FACEDET | mid.DO_FACE_SCORE | mid.DO_RAISEHANDCLS

        with APipeWrap(todo0) as pipe:
            task = ATask(todo=todo0, inpdata=tuple(images), userdata={})
            pipe.post_task(task)
            task = pipe.wait()
            faces = task.data["facedet_result_face"]
            landmarks = task.data["facedet_result_landmark"]
            for i, r in enumerate(faces):
                img0 = images[i]
                for j in range(len(r)):
                    x1, y1, x2, y2, score = r[j]
                    cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    for k in range(5):
                        cv2.circle(img0, (int(landmarks[i][j, k*2+0]), int(landmarks[i][j, k*2+1])), 2, (0, 0, 255), -1)
                
                # cv2
                cv2.imshow("image", img0)
                cv2.waitKey(0)


if __name__ == "__main__":
    unittest.main()
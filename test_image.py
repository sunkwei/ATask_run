''' 测试用例，测试图像任务
'''

from unittest import TestCase
import unittest
from atask_run.apipe import ATask, APipe, APipeWrap
import atask_run.model_id as mid
import cv2
import logging
from atask_run.timeused import TimeUsed
from atask_run.uty_postprocess import debug_draw_faceori_box
import threading

logger = logging.getLogger(__name__)

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
        
        todo0 = \
            mid.DO_ACT | \
            mid.DO_FACEDET | \
            mid.DO_FACE_SCORE | \
            mid.DO_RAISEHANDCLS | \
            mid.DO_FACEREC | \
            mid.DO_FACEORI | \
            0

        with APipeWrap(todo0, debug=False) as pipe:
            task = ATask(todo=todo0, inpdata=tuple(images), userdata={})
            pipe.post_task(task)
            task = pipe.wait()

            if mid.DO_FACEDET & todo0:
                faces = task.data["facedet_result_face"]
                landmarks = task.data["facedet_result_landmark"]

                if mid.DO_ACT & todo0:
                    ## 绘制行为
                    for b, act in enumerate(task.data["act_result"]):
                        img0 = images[b]
                        for j in range(len(act)):
                            x1, y1, x2, y2, score, cid = act[j]
                            cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
                        
                for b, r in enumerate(faces):
                    img0 = images[b]
                    for j in range(len(r)):
                        x1, y1, x2, y2, score = r[j]
                        cv2.rectangle(img0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        for k in range(5):
                            cv2.circle(img0, (int(landmarks[b][j, k*2+0]), int(landmarks[b][j, k*2+1])), 2, (0, 0, 255), -1)
                    
                    if todo0 & mid.DO_FACEORI:
                        ## 绘制 68 特征点
                        pts68 = task.data["faceori_pts68"][b]  ## (n, 68, 2)
                        for j in range(len(r)):
                            for k in range(68):
                                cv2.circle(img0, (int(pts68[j, k, 0]), int(pts68[j, k, 1])), 1, (0, 0, 255), -1)

                        ## 绘制人脸朝下 3D box
                        data = {
                            "faceori_camera_matrix": task.data["faceori_camera_m"][b],
                            "faceori_dist_coeefs": task.data["faceori_camera_coeefs"][b],
                            "facedet_result_face": task.data["facedet_result_face"][b],
                            "faceori_rvec": task.data["faceori_rvec"][b],
                            "faceori_tvec": task.data["faceori_tvec"][b],
                        }
                        debug_draw_faceori_box(img0, data)

                    # cv2.imshow("image", img0)
                    # cv2.waitKey(0)

    def test_all_images(self):
        from pathlib import Path
        P = Path("picture")
        fnames = [ str(p) for p in P.glob("*.jpg") ]
        logger.info("there are {} pictures".format(len(fnames)))

        todo0 = \
            mid.DO_ACT | \
            mid.DO_FACEDET | \
            mid.DO_FACE_SCORE | \
            mid.DO_RAISEHANDCLS | \
            mid.DO_FACEREC | \
            mid.DO_FACEORI | \
            0
        
        with APipeWrap(todo0) as pipe:
            def wait_proc():
                count = 0
                while 1:
                    pipe.wait()
                    count += 1
                    if count == len(fnames):
                        break
            th = threading.Thread(target=wait_proc)
            th.start()

            with TimeUsed("test all image"):
                for i in range(len(fnames)):
                    img = cv2.imread(fnames[i])
                    task = ATask(todo=todo0, inpdata=img, userdata={"fname": fnames[i]})
                    pipe.post_task(task)

                th.join()

    def __test_images(self, batch_size=1):
        from pathlib import Path
        P = Path("picture")
        fnames = [ str(p) for p in P.glob("*.jpg") ]
        logger.info("there are {} pictures".format(len(fnames)))

        todo0 = \
            mid.DO_ACT | \
            mid.DO_FACEDET | \
            mid.DO_FACE_SCORE | \
            mid.DO_RAISEHANDCLS | \
            mid.DO_FACEREC | \
            mid.DO_FACEORI | \
            0
        
        with APipeWrap(todo0) as pipe:
            def wait():
                for i in range(len(fnames)):
                    yield pipe.wait()

            with TimeUsed(f"test batch size {batch_size}"):
                for i in range(0, len(fnames), batch_size):
                    batch = fnames[i : i + batch_size]
                    tasks = [ ATask(todo=todo0, inpdata=cv2.imread(fname), userdata={"fname": fname}) for fname in batch ]
                    for task in tasks:
                        pipe.post_task(task)

if __name__ == "__main__":
    unittest.main()
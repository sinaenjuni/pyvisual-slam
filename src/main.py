from settings import KITTI_Monocular_dataset, Sensor_type, State
import cv2
import numpy as np
import numpy.linalg as la

from tracker import Tracker
from geometry import estimate_pose_with_fundamental, triangulate_normalized_points, estimate_pose_with_essential
from misc import to_homogeneous
from initializer import Initializer
from optimizer import Optimizer
from camera import Camera
# from feature1 import Feature_extractor_factory
from feature import FAST_ORB, FAST_GFTTD
from matcher import Matcher

from view3d import Viewer3D
from map import Frame, Map, Map_point
import misc

import sys
np.set_printoptions(threshold=sys.maxsize)
import copy


def main():
    settings = KITTI_Monocular_dataset()
    camera = Camera(settings.K, settings.width, settings.height, settings.Rt)
    feature_extractor = FAST_ORB(camera, 2000, 1.2, 8, 10)
    # feature_extractor = FAST_GFTTD()
    matcher = Matcher(cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False))

    optimizer = Optimizer()
    map = Map(settings, optimizer)
    initializer = Initializer(settings, matcher, map)
    viewer3d = Viewer3D()

    frame_ref = None
    cap = cv2.VideoCapture(settings.img_path)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    state = State.INIT_YET
    #main loop
    while 1:
        ret, img = cap.read()
        if not ret: 
            raise Exception("Fail to load image") 
        if img.ndim != 3:
            frame_view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            frame_view = img.copy()

        frame_cur = Frame(img, camera)
        feature_extractor.detect_and_compute(frame_cur)

        if frame_ref is None:
            frame_ref = frame_cur
            continue

        if state == State.INIT_YET:
            ret = initializer.init(frame_cur, frame_ref)
            if ret == State.CLOSED_IMG:
                continue
            if ret == State.INVALID_IMG:
                frame_cur = frame_ref
                frame_ref = None
                continue
            state = ret

        else:
            idx_ref = matcher.find_ref_frame(frame_cur)
            frame_ref = map.key_frames[idx_ref]
            frame_cur.set_Tcw(frame_ref.Tcw)
            # mask = np.array([v for v in frame_ref.points3d.values()])
            mask, points_3d = zip(*frame_ref.points3d.items())
            mask = np.fromiter(mask, dtype=np.uint16)
            mask_cur, mask_ref = matcher.feature_matching(
                frame_cur, frame_ref, None, mask)
            
            for mc, mr in zip(mask_cur, mask_ref):
                frame_cur.points3d[mc] = frame_ref.points3d[mr]
            
            ret = optimizer.pose_optimization(frame_cur)
            if ret:
                map.add_key_frame(frame_cur)

            
            # mask_cur, mask_ref = misc.check_cos_parallax(
                # frame_cur, frame_ref, 
                # mask_cur, mask_ref)
            # print(bad_points)

            # misc.draw_img_points(frame_ref.kps[mask_ref], frame_view, (255,0,0))
            # misc.draw_img_points(frame_cur.kps[mask_cur], frame_view, (0,255,0))
            # misc.draw_img_lines(frame_ref.kps[mask_ref], frame_cur.kps[mask_cur], frame_view)
            # misc.draw_img_lines(frame_ref.kps[mask_ref], frame_cur.kps[mask_cur], frame_view)
            # cv2.imshow("frame", frame_view)
            # key = cv2.waitKey()



        viewer3d.draw_map(map)



        # for c, r in zip(frames[-1].idx_match, frames[-2].idx_match):
        #     ref_pt = np.array(frames[-2].kps[r], np.int32)
        #     cur_pt = np.array(frames[-1].kps[c], np.int32)
        #     # print(ref_pt)
        #     cv2.line(frame_view, ref_pt, cur_pt, (0,255,0), 2, cv2.LINE_AA)
        
        # for idx in range(frames.__len__()-1):
        # frame_cur = frames[-1]
        # frame_ref = frames[-2]
        # for c, r in zip(idx_inlier_cur, idx_inlier_ref):
            # ref_pt = np.array(frame_ref.kps[r], np.int32)
            # cur_pt = np.array(frame_cur.kps[c], np.int32)
            # print(ref_pt)
            # cv2.line(frame_view, ref_pt, cur_pt, (0,255,0), 2, cv2.LINE_AA)

        # frame_ref = frame
        # cv2.drawKeypoints(frame_view, kps, frame_view)


        cv2.imshow("frame", frame_view)
        # print(FPS)
        key = cv2.waitKey()
        if key == 32: key = cv2.waitKey()
        if key == ord('q') or key == 66: break




if __name__ == '__main__':
    main()
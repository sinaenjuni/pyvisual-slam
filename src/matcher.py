import numpy as np
from map import Frame
import cv2

class Matcher:
    def __init__(self, matcher:cv2.BFMatcher):
        self.matcher = matcher

        # self.n_features = n_features
        # self.n_levels = n_levels
        # self.scale_factor = scale_factor

        # self.scale_factors = np.array([1.0] * self.n_levels)
        # sigma2_by_level = [1.0] * self.n_levels
        # self.sigma2_by_level_inv = np.array([1.0] * self.n_levels)

        # for i in range(1, self.n_levels):
        #     self.scale_factors[i] = self.scale_factors[i - 1] * self.scale_factor
        #     sigma2_by_level[i] = self.scale_factors[i] ** 2
        #     self.sigma2_by_level_inv[i] = 1.0 / sigma2_by_level[i]
        # The reason for using the inverse of sigma is the inaccuracy of the feature pyramid method. 
        # because the higher layer has a lower resolution, it has a lower accuracy.
        # Therefore the inverse of sigma in a high layer is low value.
    def add_frame(self, frame:Frame):
        self.matcher.add([frame.des])
    
    def find_ref_frame(self, frame:Frame):
        matches = self.matcher.match(frame.des)
        _, count = np.unique([m.imgIdx for m in matches], return_counts=True)
        return np.argmax(count)

    def feature_matching(self, frame_cur, frame_ref, mask_cur=None, mask_ref=None):
        des_cur = frame_cur.des if mask_cur is None else frame_cur.des[mask_cur]
        des_ref = frame_ref.des if mask_ref is None else frame_ref.des[mask_ref]

        matches = self.matcher.knnMatch(des_cur, des_ref, 2)
        
        idx_cur, idx_ref = [], []
        duplications = {}
        for m, n in matches:
            if m.distance > n.distance*0.75:
                continue
            if m.trainIdx not in duplications.keys():
                idx_cur.append(m.queryIdx)
                idx_ref.append(m.trainIdx)
                duplications[m.trainIdx] = (len(idx_ref)-1, m.distance)
            else:
                if duplications[m.trainIdx][1] > m.distance:
                    target_idx = duplications[m.trainIdx][0]
                    idx_cur[target_idx] = m.queryIdx
                    idx_ref[target_idx] = m.trainIdx
                    duplications[m.trainIdx] = (target_idx, m.distance)

        idx_cur = np.array(idx_cur) if mask_cur is None else mask_cur[np.array(idx_cur)]
        idx_ref = np.array(idx_ref) if mask_ref is None else mask_ref[np.array(idx_ref)]
        return idx_cur, idx_ref

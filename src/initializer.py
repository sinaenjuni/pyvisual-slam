import numpy as np
import numpy.linalg as la
from matcher import Matcher
from map import Frame, Map
from geometry import triangulate_normalized_points, estimate_pose_with_essential
from misc import to_homogeneous
from settings import State

class Initializer:
    def __init__(self, settings, matcher:Matcher, map:Map):
        self.pass_close_frame = True
        self.settings = settings
        self.matcher = matcher
        self.map = map

    def init(self, frame_cur:Frame, frame_ref:Frame):
        if (frame_cur.id - frame_ref.id) < 1 or not self.pass_close_frame:
            # if initial frames are close, initial pose result is bed.
            # as a result 3d points are too bed.
            print(f"Two frames are close. frame_cur's id is {frame_cur.id}, frame_ref's id is {frame_ref.id}")
            return State.CLOSED_IMG
        
        idx_matches_cur, idx_matches_ref = self.matcher.feature_matching(frame_cur, frame_ref)
        # num_inlier, R, t, pose_mask1, pose_mask2 = estimate_pose_with_fundamental(frame_cur, frame_ref, idx_matches_cur, idx_matches_ref)
        num_inlier, Tcw, pose_mask = estimate_pose_with_essential(frame_cur, frame_ref, idx_matches_cur, idx_matches_ref)
        # R, t is relative position between frame_ref and frame_cur. 
        print(f"The number of matches: {num_inlier}")

        if num_inlier < 100:
            frame_ref = frame_cur
            return State.INVALID_IMG
        
        frame_cur.set_Tcw(Tcw)
        idx_inlier_cur = idx_matches_cur[pose_mask]
        idx_inlier_ref = idx_matches_ref[pose_mask]

        points3d, pts3d_mask = triangulate_normalized_points(frame_cur, frame_ref, idx_inlier_cur, idx_inlier_ref)
        print(f"The number of 3d points: {len(pts3d_mask)}")

        self.map.add_points3d(points3d, frame_cur, frame_ref, idx_inlier_cur, idx_inlier_ref)
        self.map.add_key_frame(frame_ref)
        self.map.add_key_frame(frame_cur)

        self.matcher.add_frame(frame_ref)
        self.matcher.add_frame(frame_cur)

        # self.map.BundleAdjustment(map, 10)
        self.map.ba(10)

        # only monocular data
        # desired_median_depth = 15
        median_depth = frame_cur.compute_median_depth(self.map.points3d)
        depth_scale = self.settings.desired_median_depth / median_depth
        
        for map_point in self.map.points3d:
            pos = map_point.get_pos()
            pos = pos * depth_scale
            map_point.update_pos(pos) 

        tcw = frame_cur.tcw * depth_scale
        frame_cur.set_tcw(tcw)

        return State.INIT
import numpy as np
import numpy.linalg as la
from collections import deque
from ordered_set import OrderedSet

from misc import to_homogeneous
from optimizer import Optimizer

class Map:
    def __init__(self, settings, optimizer:Optimizer):
        self.settings = settings
        self.optimizer = optimizer

        self.key_frames = OrderedSet()
        self.points3d = set()

    def add_points3d(self, points3d, frame_cur, frame_ref, mask_cur, mask_ref):
        uv_cur, uv_depth_cur = frame_cur.proj_points(points3d)
        bad_depth_cur = uv_depth_cur <= 0
        uv_ref, uv_depth_ref = frame_ref.proj_points(points3d)
        bad_depth_ref = uv_depth_ref <= 0

        # compute back-projected rays (unit vectors)
        rays1 = (frame_cur.Rwc @ to_homogeneous(frame_cur.kpsn[mask_cur]).T).T
        norm_rays1 = la.norm(rays1, axis=-1, keepdims=True)
        rays1 /= norm_rays1

        rays2 = (frame_ref.Rwc @ to_homogeneous(frame_ref.kpsn[mask_ref]).T).T
        norm_rays2 = la.norm(rays2, axis=-1, keepdims=True)  
        rays2 /= norm_rays2

        # compute dot products of rays
        cos_parallax = np.sum(rays1 * rays2, axis=1)
        # bad_cos_parallax = np.logical_and(np.logical_or(cos_parallax < 0, cos_parallax > cos_max_parallax), np.logical_not(recovered3d_from_stereo))           
        bad_cos_parallax = np.logical_or(cos_parallax < 0, cos_parallax > self.settings.cos_max_parallax)

        # compute reprojection errors and check chi2
        errs1 = uv_cur - frame_cur.kps[mask_cur]
        # squared reprojection errors 
        errs1_sqr = np.sum(errs1 * errs1, axis=1)  
        # kps1_levels = frame_cur.octave[idx_inlier_cur]
        # invSigmas2_1 = feature_extractor.scale2inv_factors[kps1_levels] 
        invSigmas2_1 = frame_cur.scale2inv[mask_cur]
        chis2_1_mono = errs1_sqr * invSigmas2_1         # chi square 
        bad_chis2_1 = chis2_1_mono > self.settings.kChi2Mono

        # compute reprojection errors and check chi2
        errs2 = uv_ref - frame_ref.kps[mask_ref]
        # squared reprojection errors 
        errs2_sqr = np.sum(errs2 * errs2, axis=1)  
        # kps2_levels = frame_ref.octave[idx_inlier_ref]
        # invSigmas2_2 = feature_extractor.scale2inv_factors[kps2_levels] 
        invSigmas2_2 = frame_ref.scale2inv[mask_ref]
        chis2_2_mono = errs2_sqr * invSigmas2_2         # chi square 
        bad_chis2_2 = chis2_2_mono > self.settings.kChi2Mono

        ratio_scale_consistency = self.settings.scale_consistency_factor * self.settings.scale_factor
        # scale_factors_x_depths1 =  feature_extractor.scale2_factors[kps1_levels] * uv_depth_cur
        scale_factors_x_depths1 =  frame_cur.scale2[mask_cur] * uv_depth_cur
        scale_factors_x_depths1_x_ratio_scale_consistency = scale_factors_x_depths1 * ratio_scale_consistency                             
        # scale_factors_x_depths2 =  feature_extractor.scale2_factors[kps2_levels] * uv_depth_ref   
        scale_factors_x_depths2 =  frame_ref.scale2[mask_ref] * uv_depth_ref   
        scale_factors_x_depths2_x_ratio_scale_consistency = scale_factors_x_depths2 * ratio_scale_consistency      
        
        bad_scale_consistency = np.logical_or( (scale_factors_x_depths1 > scale_factors_x_depths2_x_ratio_scale_consistency), 
                                                (scale_factors_x_depths2 > scale_factors_x_depths1_x_ratio_scale_consistency) )   
        bad_points = bad_cos_parallax | bad_depth_cur | bad_depth_ref | bad_chis2_1 | bad_chis2_2 | bad_scale_consistency


        for point3d, is_bad, idx_cur, idx_ref in zip(points3d, bad_points, mask_cur, mask_ref):
            if is_bad:
                continue
            map_point = Map_point()
            map_point.update_pos(point3d)
            map_point.des = frame_cur.des[idx_cur]
            map_point.set_observation(frame_cur, idx_cur)
            map_point.set_observation(frame_ref, idx_ref)
            self.points3d.add(map_point)
            frame_ref.points3d[idx_ref] = map_point
            frame_cur.points3d[idx_cur] = map_point
            
        # bad_points_mask = np.where(bad_points==False)[0]
        # frame_ref.tracked_points_mask = mask_ref[bad_points_mask]
        # frame_cur.tracked_points_mask = mask_cur[bad_points_mask]
        
    def add_key_frame(self, frame):
        self.key_frames.append(frame)

    def ba(self, n_iters=10):
        self.optimizer.BundleAdjustment(self, n_iters)

class Frame:
    id = -1
    def __init__(self, img, camera):
        Frame.id = Frame.id + 1
        self.id = Frame.id
        self.img = img
        self.camera = camera

        self.kps = []
        self.kps_r = []
        self.kpsn = [] # normalized plan points
        self.des = []
        self.des_r = []
        self.octave = []
        self.octave_r = []
        self.scale2 = []
        self.scale2inv = []
        self.tracked_points_mask = []
        self.points3d = {}
        
        self.Tcw = np.eye(4, dtype=np.float64)
        self.Twc = np.eye(4, dtype=np.float64)
        # Tcw means to convert the world points to the camera ones
        self.Rcw = self.Tcw[:3, :3]
        self.tcw = self.Tcw[:3, [3]]
        self.Rwc = self.Rcw.T
        self.twc = -self.Rcw.T @ self.tcw
        self.Ow = -(self.Rwc @ self.tcw)

    def set_Tcw(self, Tcw):
        # Translate camera coordinates to world ones.
        self.Tcw = Tcw
        self.Rcw = self.Tcw[:3,:3]
        self.tcw = self.Tcw[:3, [3]]

        # Translate world coordinates to camera ones.
        # image_points = K @ Tcw @ world_points
        self.Rwc = self.Rcw.T
        self.twc = -self.Rcw.T @ self.tcw
        self.Twc[:3, :3] = self.Rwc
        self.Twc[:3, [3]] = self.twc
        # translate world points to camera points. its means position of current frame.
        self.Ow = -(self.Rwc @ self.tcw) # Origin of a camera coordinate based on world coordinate.

    def set_tcw(self, tcw):
        self.Tcw[:3, [3]] = tcw
        self.Rcw = self.Tcw[:3,:3]
        self.tcw = self.Tcw[:3, [3]]
        
        self.Rwc = self.Rcw.T
        self.twc = -self.Rcw.T @ self.tcw
        self.Twc[:3, :3] = self.Rwc
        self.Twc[:3, [3]] = self.twc
        self.Ow = -(self.Rwc @ self.tcw)

    def proj_points(self, points):
        framed_pts = (self.Rcw @ points.T + self.tcw).T
        proj_pts, depth = self.camera.project(framed_pts)
        return proj_pts, depth

    def compute_median_depth(self, points):
        kps = np.array([p.pos for p in points])
        z = (self.Rcw[2,:3] @ kps.T) + self.tcw[2]
        z = np.sort(z)
        return z[(len(z)-1)//2]
        
class Map_point:
    id = -1
    def __init__(self):
        Map_point.id += 1
        self.id = Map_point.id
        self.pos = np.empty(3, dtype=np.float64)
        self.des = np.empty(32, dtype=np.uint8)
        self._observation = dict()
        self.color = None

    def get_id(self):
        return self.id

    def get_pos(self):
        return self.pos
    def get_homo_pos(self):
        return np.hstack([self.pos, 1])
    def get_desc(self):
        return self.des
    def set_color(self,c):
        self.c = c
    def get_color(self):
        return self.c
    
    def update_pos(self, new_pos):  # memory referencecalc_pose_with_F issue
        self.pos[0] = new_pos[0]
        self.pos[1] = new_pos[1]
        self.pos[2] = new_pos[2]

    # def add_observation(self, frame, kp_idx):
    #     # if frame_idx not in self.observation.keys():
    #         # self.observation[frame_idx] = [] 
    #     # self.observation[frame_idx].append(kp_idx)
    #     self.observation.append((frame, kp_idx))

    def get_observation(self):
        return list(self._observation.items())
    
    def set_observation(self, frame, idx_uv_point):
        self._observation[frame] = idx_uv_point
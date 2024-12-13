import cv2
import numpy as np

from map import Frame
from misc import to_homogeneous

def estimate_pose_with_essential(frame_cur:Frame, frame_ref:Frame, mask_cur, mask_ref):
    matches_cur = frame_cur.kpsn[mask_cur]
    matches_ref = frame_ref.kpsn[mask_ref]
    # Calculate translation matrix from frame_cur to frame_ref.
    # [R|t] is Twc that convert The camera coordinates of frame_cur to the world ones of frame_ref.
    # E, inlier1 = cv2.findEssentialMat(matches_cur, matches_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.0004)
    E, pose_mask = cv2.findEssentialMat(matches_ref, matches_cur, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.0004)
    num_inlier, R, t, inlier2 = cv2.recoverPose(E=E, points1=matches_ref, points2=matches_cur, focal=1, pp=(0., 0.))
    # [R,|t] is homogeneous transformation matrix with respect to 'ref' frame, pr_= Trc * pc_   
    Tcw = np.eye(4)
    Tcw[:3, :3] = R
    Tcw[:3, [3]] = t
    
    pose_mask = np.where(pose_mask.squeeze()==1)[0]
    return num_inlier, Tcw, pose_mask

def estimate_pose_with_fundamental(frame_cur:Frame, frame_ref:Frame, match_idx_cur, match_idx_ref):
    K = frame_cur.camera.K
    matches_cur = frame_cur.kps[match_idx_cur]
    matches_ref = frame_ref.kps[match_idx_ref]

    # The order of points1 and 2 is important
    F, inlier1 = cv2.findFundamentalMat(
                points1=matches_cur, 
                points2=matches_ref,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=0.1, # reprojection pixel error
                confidence=0.999, # N = \log(1 - \text{confidence}) \, / \, \log(1 - w^s), the number of iterations
                # maxIters=1
                )
    E = K.T @ F @ K

    num_inlier, R, t, inlier2 = cv2.recoverPose(E=E,
                                points1=matches_cur, 
                                points2=matches_ref, 
                                cameraMatrix=K)
    return num_inlier, R, t, inlier1.squeeze(), inlier2.squeeze()


def triangulate_normalized_points(frame_cur:Frame, frame_ref:Frame, mask_cur, mask_ref):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has
    P1 = frame_cur.Tcw[:3] # [R1w, t1w]
    P2 = frame_ref.Tcw[:3] # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1, P2, frame_cur.kpsn[mask_cur].T, frame_ref.kpsn[mask_ref].T)
    good_mask = np.where(point_4d_hom[3]!= 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3]

    # if __debug__:
    #     if False: 
            # point_reproj = P1 @ point_4d;
            # point_reproj = point_reproj / point_reproj[2] - to_homogeneous(frame_cur.kpsn[match_idx_cur]).T
            # err = np.sum(point_reproj**2)
            # print('reproj err: ', err)     

            # point_reproj = P2 @ point_4d;
            # point_reproj = point_reproj / point_reproj[2] - to_homogeneous(frame_ref.kpsn[match_idx_ref]).T
            # err = np.sum(point_reproj**2)
            # print('reproj err: ', err)     
    points_3d = point_4d[:3, :].T
    return points_3d, good_mask  


def triangulate_points(frame_cur:Frame, frame_ref:Frame, match_idx_cur, match_idx_ref):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has
    K = frame_cur.K
    P1 = K@frame_cur.Tcw[:3] # [R1w, t1w]
    P2 = K@frame_ref.Tcw[:3] # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1, P2, frame_cur.kps[match_idx_cur].T, frame_ref.kpn[match_idx_ref].T)
    good_mask = np.where(point_4d_hom[3]!= 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3]

    # if __debug__:
    #     if False: 
    point_reproj = P1 @ point_4d;
    point_reproj = point_reproj / point_reproj[2] - to_homogeneous(frame_cur.kpsn[match_idx_cur]).T
    err = np.sum(point_reproj**2)
    print('reproj err: ', err)     

    point_reproj = P2 @ point_4d;
    point_reproj = point_reproj / point_reproj[2] - to_homogeneous(frame_ref.kpsn[match_idx_ref]).T
    err = np.sum(point_reproj**2)
    print('reproj err: ', err)     

    # return point_4d.T
    points_3d = point_4d[:3, :].T
    return points_3d, good_mask  
import numpy as np
import numpy.linalg as la
from misc import to_homogeneous

class Camera:
    # To exists only one instance, I used the Singleton pattern.
    # __instance = None
    # __initialized = False
    # def __new__(cls, *args, **kwargs):
    #     if not cls.__instance:
    #         cls.__instance = super().__new__(cls)
    #     return cls.__instance

    def __init__(self, K, width, height, Rt):
        # if not Camera.__initialized:
            self.K = K
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]

            self.Kinv = la.inv(self.K)
            self.width = width
            self.height = height
            self.Rt = Rt
            # Camera.__initialized = True
    
    def unproject(self, points):
        # in: A image points of shape [Nx2]
        # out: A normalized input image points [Nx2]
        return (self.Kinv @ to_homogeneous(points).T)[:2].T

    def project(self, points):
        # in: project the 3D points or the array of 3D points (w.r.t. camera frame), of shape [Nx3]
        # out: [Nx2] image points, [Nx1] array of map point depths
        # u = self.fx * xc[0]/xc[2] + self.cx
        # v = self.fy * xc[1]/xc[2] + self.cy  
        proj_pts = self.K @ points.T
        zs = proj_pts[2]
        proj_pts = proj_pts[:2] / zs
        return proj_pts.T, zs
    
if __name__ == "__main__":
    from settings import KITTI_Monocular_dataset
    settings = KITTI_Monocular_dataset()
    camera1 = Camera(K=settings.K, width=settings.width, height=settings.height, Rt=settings.Rt)
    # camera2 = Camera()
    # print(camera1 is camera2)
    print(camera1.K)

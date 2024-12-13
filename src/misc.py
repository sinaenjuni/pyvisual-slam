import enum
from dataclasses import dataclass
import numpy as np
import numpy.linalg as la
import cv2

def to_homogeneous(input):
    if input.ndim == 1:
        return np.hstack([input, 1])
    else:
        return np.hstack((input, np.ones((len(input), 1))))

def draw_img_points(points, out_img, color=(255,0,0)):
    for point in points.astype(np.uint32):
        print(point)
        cv2.circle(out_img, point, 5, color, -1, cv2.LINE_AA)

def draw_img_lines(points_cur, points_ref, out_img, color=(255,0,0)):
    for point_cur, point_ref in zip(points_cur.astype(np.uint32), points_ref.astype(np.uint32)):
        cv2.circle(out_img, point_cur, 5, (255,0,0), -1, cv2.LINE_AA)
        cv2.circle(out_img, point_ref, 5, (0,255,0), -1, cv2.LINE_AA)
        cv2.line(out_img, point_cur, point_ref, (0,0,255), 2, cv2.LINE_AA)

def check_cos_parallax(frame_cur, frame_ref, mask_cur=None, mask_ref=None, cos_max_parallax=0.99998):
    kpsn_cur = frame_cur.kpsn if mask_cur is None else frame_cur.kpsn[mask_cur]
    kpsn_ref = frame_ref.kpsn if mask_ref is None else frame_ref.kpsn[mask_ref]

    # compute back-projected rays (unit vectors)
    rays1 = (frame_cur.Rwc @ to_homogeneous(kpsn_cur).T).T
    norm_rays1 = la.norm(rays1, axis=-1, keepdims=True)
    rays1 /= norm_rays1

    rays2 = (frame_ref.Rwc @ to_homogeneous(kpsn_ref).T).T
    norm_rays2 = la.norm(rays2, axis=-1, keepdims=True)  
    rays2 /= norm_rays2

    # compute dot products of rays
    cos_parallax = np.sum(rays1 * rays2, axis=1)
    # bad_cos_parallax = np.logical_and(np.logical_or(cos_parallax < 0, cos_parallax > cos_max_parallax), np.logical_not(recovered3d_from_stereo))           
    # bad_cos_parallax = np.logical_or(cos_parallax < 0, cos_parallax > self.settings.cos_max_parallax)
    # bad_cos_parallax = np.logical_or(cos_parallax < 0, cos_parallax > cos_max_parallax)
    good_cos_parallax = np.logical_and(cos_parallax > 0.9999, cos_parallax < cos_max_parallax)
    
    return mask_cur[good_cos_parallax], mask_ref[good_cos_parallax]

# def inv_pose(pose):
    # R, t = pose[:3,:3], pose[:3, 3]
    


# data = np.random.randint(0, 100, (10, 2)).astype(np.float32)
# print(to_homogeneous(data))


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))

    def get_neighbors(self, node):
        return self.graph.get(node, [])

    def __str__(self):
        result = ""
        for node in self.graph:
            result += f"{node} -> {self.graph[node]}\n"
        return result


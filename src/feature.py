import cv2
import numpy as np
import heapq
from map import Frame
from camera import Camera
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

class Feature_extractor(ABC):
    @abstractmethod
    def detect(self):
        pass
    @abstractmethod
    def compute(self):
        pass
    @abstractmethod
    def detect_and_compute(self):
        pass


class FAST_GFTTD(Feature_extractor):
    def __init__(self):
        n_features = 4000
        self.n_levels = 8
        self.scale_factor = 1.2
        self.scale2_factors = np.array([self.scale_factor ** i for i in range(self.n_levels)])
        self.scale2inv_factors = np.array([1.0/scale_factor for scale_factor in self.scale2_factors])

        self.detector = cv2.GFTTDetector.create(n_features, 0.01, 5)
        self.descriptor = cv2.ORB.create()
        matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, img):
        points = self.detector.detect(img, None)
        return points
    
    def compute(self, img, points):
        points, des = self.descriptor.compute(img, points)
        return points, des
    
    def detect_and_compute(self, frame:Frame):
        points = self.detect(frame.img)
        points, des = self.compute(frame.img, points)

        frame.kps = np.array([p.pt for p in points])
        frame.kpsn = frame.camera.unproject(frame.kps)
        frame.des = des
        frame.octave = np.array([p.octave for p in points])
        frame.scale2 = self.scale2_factors[frame.octave]
        frame.scale2inv = self.scale2inv_factors[frame.octave]
        

class Octree_node:
    def __init__(self, points, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        mask = np.bitwise_and(
            np.bitwise_and(self.xmin <= points[:, 0], self.xmax > points[:, 0]),
            np.bitwise_and(self.ymin <= points[:, 1], self.ymax > points[:, 1])
        )
        self.points = points[mask]

        is_depth = (self.xmax - self.xmin) < 3 or (self.ymax - self.ymin) < 3
        is_point = len(self.points) == 1
        self.is_done = is_depth or is_point
        
    def divide_node(self):
        xhalf = self.xmin + ((self.xmax - self.xmin) // 2)
        yhalf = self.ymin + ((self.ymax - self.ymin) // 2)

        UL = Octree_node(self.points, self.xmin, self.ymin, xhalf, yhalf)
        UR = Octree_node(self.points, xhalf, self.ymin, self.xmax, yhalf)
        BL = Octree_node(self.points, self.xmin, yhalf, xhalf, self.ymax)
        BR = Octree_node(self.points, xhalf, yhalf, self.xmax, self.ymax)

        UL = None if len(UL) == 0 else UL
        UR = None if len(UR) == 0 else UR
        BL = None if len(BL) == 0 else BL
        BR = None if len(BR) == 0 else BR
        return UL, UR, BL, BR

    def __len__(self):
        return len(self.points)
    
    def __lt__(self, other):
        return len(self) > len(other)

    def __str__(self):
        return f"{id(self)} {self.xmin}, {self.ymin}, {self.xmax}, {self.ymax} {len(self)}"
    # def __repr__(self):
        # return f"{id(self)} {self.UL}, {self.UR}, {self.BL}, {self.BR}"
        # return f"{id(self)} {self.minx}, {self.miny}, {self.maxx}, {self.maxy}"
    def draw_data(self, img, only_box=False, color=(255,0,0), thickness=1):
        img = cv2.rectangle(img, (self.xmin, self.ymin), (self.xmax, self.ymax), color, thickness)
        if only_box:
            for x, y, response, octave in self.points:
                center = list(map(int, [x, y]))
                cv2.circle(img, center, 2, color, -1 * thickness, cv2.LINE_AA)
        return img

class FAST_ORB(Feature_extractor):
    def __init__(self, camera:Camera, n_points=2000, scale_factor=1.2, n_levels=8, feature_size=10, margin=5):
        self.detector = cv2.FastFeatureDetector.create(threshold=20, nonmaxSuppression=True)
        self.descriptor = cv2.ORB.create()
        self.camera = camera
        self.w = camera.width
        self.h = camera.height

        self.n_points = n_points
        self.n_levels = n_levels
        self.feature_size = feature_size
        self.margin = margin
        self.scale_factor = scale_factor
        self.scale2_factors = np.array([self.scale_factor ** i for i in range(self.n_levels)])
        self.scale2inv_factors = np.array([1.0/scale_factor for scale_factor in self.scale2_factors])
        # The reason for using the inverse of sigma is the inaccuracy of the feature pyramid method. 
        # because the higher layer has a lower resolution, it has a lower accuracy.
        # Therefore the inverse of sigma in a high layer is low value.
        self.imgs = [None] * self.n_levels
        self.points = [] 


    def detect(self, img):
        self.make_image_pyramid(img)
        self.detect_feature_all_octave()
        
        if len(self.points) > self.n_points:
            points = self.octree_filtering(0, 0, self.w, self.h)
            return points
        else:
            return self.points

    def compute(self, img, points):
        cv_points = [cv2.KeyPoint(x, y, self.feature_size, -1, response, int(octave), -1) for x, y, response, octave in points]
        points, des = self.descriptor.compute(img, cv_points)
        return points, des
    
    def detect_and_compute(self, frame:Frame):
        points = self.detect(frame.img)
        points, des = self.compute(frame.img, points)
        
        frame.kps = np.array([p.pt for p in points])
        frame.kpsn = frame.camera.unproject(frame.kps)
        frame.des = des
        frame.octave = np.array([p.octave for p in points])
        frame.scale2 = self.scale2_factors[frame.octave]
        frame.scale2inv = self.scale2inv_factors[frame.octave]

        return points, des
    
    def make_image_pyramid(self, img):
        inv_scale = self.scale2inv_factors[1]
        self.imgs[0] = img
        for i in range(1, self.n_levels):
            self.imgs[i] = cv2.resize(self.imgs[i-1], (0,0), fx=inv_scale, fy=inv_scale)

    def detect_feature_with_octave(self, octave):
        points = self.detector.detect(self.imgs[octave])
        for point in points:
            x, y = point.pt
            x = x * self.scale2_factors[octave]
            y = y * self.scale2_factors[octave]
            self.points.append([x, y, point.response, octave])

    def detect_feature_all_octave(self):
        self.points = [] 
        # for i in range(self.n_levels):
            # self.detect_feature_with_octave(i)
            
        futures = []
        with ThreadPoolExecutor(max_workers = 4) as executor:
            for i in range(self.n_levels):
                futures.append(
                    executor.submit(
                        self.detect_feature_with_octave, i))
            wait(futures) # wait all the task are completed 
        self.points = np.array(self.points)

    def octree_filtering(self, xmin, ymin, xmax, ymax):
        n_init_node = round((xmax - xmin) / (ymax - ymin))

        hX = (xmax-xmin)/n_init_node
        l_nodes = []
        for i in range(n_init_node):
            node = Octree_node(self.points, round(hX*i), 0, round(hX*(i+1)), ymax - ymin)
            heapq.heappush(l_nodes, node)

        done_nodes = []
        heapq.heapify(l_nodes)
        while 1:
            node = heapq.heappop(l_nodes)
            if node.is_done:
                done_nodes.append(node)
                continue
            if len(node) > 2:
                for new_node in node.divide_node():
                    if new_node is not None:
                        heapq.heappush(l_nodes, new_node)
            if self.n_points <= len(done_nodes) + len(l_nodes):
                heapq.heappush(l_nodes, node)
                break
        
        l_nodes.extend(done_nodes)

        ret = []
        for node in l_nodes:
            best_response = 0
            best_point = None
            for point in node.points:
                if best_response <= point[2]:
                    best_response = point[2]
                    best_point = point
            ret.append(best_point)
        # for node in l_nodes:
            # color = np.random.randint(0,255, 3, dtype=np.uint8).tolist()
            # node.draw_data(img, True, color, 1)
            # ret = ret + len(node.points)

        ret.sort(key=lambda x : x[2], reverse=True)
        return np.array(ret[:self.n_points])


if __name__ == "__main__":
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    import time
    from pyinstrument import Profiler
    profiler = Profiler()

    from settings import KITTI_Monocular_dataset
    settings = KITTI_Monocular_dataset()
    camera = Camera(settings.K, settings.width, settings.height, settings.Rt)

    cap = cv2.VideoCapture(settings.img_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1131)

    fd = FAST_ORB(camera, 2000)
    while 1:
        ret, img = cap.read()
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = Frame(gray_img, camera)
        if not ret:
            break
        # start = time.perf_counter()
        # profiler.start()

        # fd.make_image_pyramid(frame.img)
        # fd.detect_feature_with_octave(0)
        # fd.detect_feature_all_octave()
        # print(fd.points)

        # points = fd.detect(gray_img)
        fd.detect_and_compute(frame)
        # print(points[0].angle)
        print(np.unique([octave for octave in frame.octave], return_counts=True))

        # end = time.perf_counter()
        # print(f"Execution time: {end - start} seconds")
        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True))
        # profiler.reset()

        # kps = fd.detect(gray_img)
        
        # img = cv2.drawKeypoints(img, points, None)
        # for x, y, response, octave in points:
            # cv2.circle(img, (int(x), int(y)), 2, (255,0,0), -1, cv2.LINE_AA)
        cv2.imshow("img", img)
        key = cv2.waitKey()
        if key == 32: key = cv2.waitKey()
        if key == ord('q'): break
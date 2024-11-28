import cv2
import numpy as np
import heapq

class Octree_node:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.is_done = False
        self.points = []

    def add_point(self, point):
        x, y = point.pt
        if not (self.xmin <= int(x) <= self.xmax):
            raise ValueError(f"The range of x [{self.xmin} - {self.xmax}] but the point range of this node is [{x, y}]")
        if not (self.ymin <= int(y) <= self.ymax):
            raise ValueError(f"The range of y [{self.ymin} - {self.ymax}] but the point range of this node is [{x, y}]")
        self.points.append(point)

    def divide_node(self):
        xhalf = self.xmin + ((self.xmax - self.xmin) // 2)
        yhalf = self.ymin + ((self.ymax - self.ymin) // 2)

        UL = Octree_node(self.xmin, self.ymin, xhalf, yhalf)
        UR = Octree_node(xhalf, self.ymin, self.xmax, yhalf)
        BL = Octree_node(self.xmin, yhalf, xhalf, self.ymax)
        BR = Octree_node(xhalf, yhalf, self.xmax, self.ymax)

        for point in self.points:
            x, y = point.pt
            if x < xhalf:
                if y < yhalf:
                    UL.add_point(point)
                else:
                    BL.add_point(point)
            else:
                if y < yhalf:
                    UR.add_point(point)
                else:
                    BR.add_point(point)
        
        if len(UL) == 0:
            UL = None
        if len(UR) == 0:
            UR = None
        if len(BL) == 0:
            BL = None
        if len(BR) == 0:
            BR = None
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
            for point in self.points:
                center = list(map(int, point.pt))
                cv2.circle(img, center, 2, color, -1 * thickness, cv2.LINE_AA)
        return img


class Feature_detector:
    def __init__(self, HEIGHT, WIDTH, n_points=2000, scale_factor=1.2, n_levels=8):
        self.detector = cv2.FastFeatureDetector.create(threshold=20, nonmaxSuppression=True)

        self.n_points = n_points
        self.n_levels = n_levels
        self.scale_factor = scale_factor
        self.scale_factors = [self.scale_factor ** i for i in range(self.n_levels)]
        self.w = WIDTH
        self.h = HEIGHT
        self.imgs = [np.empty((self.h, self.w), dtype=np.int8)] * n_levels
        self.points = []

    def detect(self, img):
        self.get_image_pyramid(img)
        for level in range(self.n_levels):
            self.detect_feature_with_level(level)
        points = self.octree_filtering(0, 0, self.w, self.h)
        return points
    
    def get_image_pyramid(self, img):
        inv_scale = 1.0/self.scale_factor
        resized_img = img
        for i in range(self.n_levels):
            self.imgs[i] = resized_img
            resized_img = cv2.resize(resized_img, (0,0), fx=inv_scale, fy=inv_scale)

    def detect_feature_with_level(self, level):
        points = self.detector.detect(self.imgs[level])
        for point in points:
            # kp.size = 
            point.octave = level
            x, y = point.pt
            x = x * self.scale_factors[level]
            y = y * self.scale_factors[level]
            point.pt = (x, y)
            self.points.append(point)

    def octree_filtering(self, xmin, ymin, xmax, ymax):
        n_init_node = round((xmax - xmin) / (ymax - ymin))

        hX = (xmax-xmin)/n_init_node
        l_nodes = []

        for i in range(n_init_node):
            node = Octree_node(round(hX*i), 0, round(hX*(i+1)), ymax - ymin)
            l_nodes.append(node)

        for kp in self.points:
            idx = int(kp.pt[0]/hX)
            l_nodes[idx].add_point(kp)

        done_count = 0
        heapq.heapify(l_nodes)
        while 1:
            node = heapq.heappop(l_nodes)
            if self.n_points <= done_count or self.n_points <= len(l_nodes):
                heapq.heappush(l_nodes, node)
                break
            if len(node) == 0:
                continue
            if len(node) == 1:
                node.is_done = True
                done_count = done_count + 1
                heapq.heappush(l_nodes, node)
                continue
            
            nodes = node.divide_node()
            for node in nodes:
                if node:
                    heapq.heappush(l_nodes, node)
        
        self.points.clear()
        for node in l_nodes:
            best_response = 0
            best_point = None
            for point in node.points:
                if best_response <= point.response:
                    best_response = point.response
                    best_point = point
            # node.points.clear()
            # node.points.append(best_point)
            self.points.append(best_point)
        
        # for node in l_nodes:
        #     color = np.random.randint(0,255, 3, dtype=np.uint8).tolist()
        #     node.draw_data(img, True, color, 1)
        #     ret = ret + len(node.points)

        self.points.sort(key=lambda x : x.response, reverse=True)
        return self.points[:self.n_points]


if __name__ == "__main__":
    import yaml
    with open("./settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    PATH = settings["KITTI"]["PATH"]["img_dirs"][0]

    cap = cv2.VideoCapture(PATH)
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape
    
    fd = Feature_detector(h, w)
    kps = fd.detect(gray_img)

    img = cv2.drawKeypoints(img, kps, None)
    cv2.imshow("img", img)
    cv2.waitKey()
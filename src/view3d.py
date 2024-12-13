import pypangolin as pango
from OpenGL.GL import *

from multiprocessing import Process, Queue, Value

import socket
import struct
import pickle

import numpy as np
import numpy.linalg as la
from map import Map

class Map_model:
    def __init__(self):
        self.points = []
        self.poses = []
        # self.K = []


class Viewer3D:
    def __new__(cls, *args):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Viewer3D, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, scale=1.0, win_width=1024, win_height=768):
        self.scale = scale
        self.win_width = win_width
        self.win_height = win_height

        self.map_queue = Queue()
        self._is_stop  = Value('i',0)
        
        self.map_state = None

        self.view_proc = Process(target=self.viewer_thread)
        self.view_proc.daemon = True
        self.view_proc.start()

    def viewer_thread(self):
        self.viewer_init()

        while not pango.ShouldQuit() and not self._is_stop.value:
            self.viewer_refresh()

    def viewer_init(self):
        pango.CreateWindowAndBind("Viewer", self.win_width, self.win_height)
        glEnable(GL_DEPTH_TEST)

        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        viewpoint_x =   0 * self.scale
        viewpoint_y = -100 * self.scale
        viewpoint_z = -0.1 * self.scale
        viewpoint_f = 1000
        # https://stackoverflow.com/questions/64743308/pangolin-s-cam-vs-d-cam-need-help-understanding-the-difference
        
        self.view_projection = pango.ProjectionMatrix(
                                self.win_width, self.win_height, viewpoint_f, viewpoint_f, 
                                self.win_width//2, self.win_height//2, 0.1, 1000)
        self.view_look_view = pango.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z,
                                0, 0, 0, 0.0, -1.0, 0.0)
        self.s_cam = pango.OpenGlRenderState(self.view_projection, self.view_look_view)
        self.handler = pango.Handler3D(self.s_cam)

        # self.s_cam = pango.OpenGlRenderState(
        #     pango.ProjectionMatrix(self.win_width, self.win_height, 
        #                         viewpoint_f, viewpoint_f, 
        #                         self.win_width//2, self.win_height//2, 0.1, 5000),
        #     pango.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z,
        #                         0, 0, 0, 0.0, -1.0, 0.0)
        #     # pango.ModelViewLookAt(0, -100, -0.1, 0, 0, 0, 0.0, -1.0, 0.0)
        # )


        # Create Interactive View in window
        self.d_cam = pango.CreateDisplay()
        self.d_cam.SetBounds(pango.Attach(0.0), pango.Attach(1.0), 
                             pango.Attach.Pix(175), pango.Attach(1.0), 
                             -self.win_width / self.win_height)
        self.d_cam.SetHandler(self.handler)

        # self.d_cam = (
        #     pango.CreateDisplay()
        #         .SetBounds(
        #             pango.Attach(0.0), 
        #             pango.Attach(1.0), 
        #             # pango.Attach.Pix(180/self.win_width), # 175 
        #             pango.Attach.Pix(180), # 175 
        #             pango.Attach(1.0), 
        #             -self.win_width / self.win_height)
        #         .SetHandler(self.handler)
        # )

    def viewer_refresh(self):
        while not self.map_queue.empty():
            self.map_state = self.map_queue.get()

        glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT)
        # glClearColor(1.0, 1.0, 1.0, 1.0)
        glPointSize(3)

        pango.glDrawAxis(1)

        if self.map_state is not None:
            if not len(self.map_state.points) == 0:
                glBegin(GL_POINTS)
                glColor3f(1.0,1.0,1.0)
                for x, y, z in self.map_state.points:
                    # glColor4f(1.0, 1.0, 1.0, 1.0)
                    # print(x, y, z)
                    glVertex3f(x, y, z)
                glEnd()

            if not len(self.map_state.poses) == 0:
                for Twc, Kinv, w, h in self.map_state.poses:
                    pango.glDrawFrustum(Kinv, w, h, Twc, 1.)

                self.s_cam.Follow(pango.OpenGlMatrix(Twc))

        self.d_cam.Activate(self.s_cam)
        pango.FinishFrame()


    def draw_map(self, map:Map):
        if self._is_stop.value:
            return False

        map_model = Map_model()

        for key_frame in map.key_frames:
            Kinv = key_frame.camera.Kinv
            Twc = key_frame.Twc
            w = key_frame.camera.width
            h = key_frame.camera.height
            map_model.poses.append([Twc, Kinv, w, h])

        for map_point in map.points3d:
            pos = map_point.get_pos()
            map_model.points.append(pos)
            
        self.map_queue.put(map_model)

    def quit(self):
        print('Viewer stopping')   
        self._is_stop.value = 1
        self.view_proc.join()
        print('Viewer stopped')   

    def run(self):
        glPointSize(2)
        while not pango.ShouldQuit():
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            pango.glDrawAxis(1)

            # if self.points is not None:
            #     glColor4f(1.0, 1.0, 1.0, 1.0)
            #     pango.glDrawPoints(self.points)
            if not len(self.map_queue) == 0:
                glBegin(GL_POINTS)
                glColor3f(1.0,1.0,1.0)
                for map_point in self.map_queue:
                    # glColor4f(1.0, 1.0, 1.0, 1.0)
                    x, y, z = map_point.get_pos()
                    # print(x, y, z)
                    glVertex3f(x, y, z)
                glEnd()
            # if self.target_points is not None:
            #     glColor4f(1.0, 0.0, 0.0, 1.0)
            #     pango.glDrawPoints(self.target_points)
            # if self.key_frames is not None:
            #     glColor4f(1.0, 1.0, 1.0, 1.0)
            #     for key_frame in self.key_frames:
            #         pango.glDrawFrustum(self.Kinv, self.WIDTH, self.HEIGHT, key_frame, 1.)
            if not len(self.key_frames) == 0:
                glColor4f(1.0, 1.0, 1.0, 1.0)
                for key_frame in self.key_frames:
                    pango.glDrawFrustum(self.Kinv, self.WIDTH, self.HEIGHT, key_frame.Twc(), 1.)
            # if self.curr_frame is not None:
            #     glColor4f(0.0, 1.0, 0.0, 1.0)
            #     pango.glDrawFrustum(self.Kinv, self.WIDTH, self.HEIGHT, self.curr_frame, 1.)
            #     self.s_cam.Follow(pango.OpenGlMatrix(self.curr_frame))

            self.d_cam.Activate(self.s_cam)
            pango.FinishFrame()
        
        self.is_closed = True


def model_server(viewer):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 1121))
    server_socket.settimeout(1.0) # set timeout
    server_socket.listen(1) # set status for receiving

    try:
        while 1:
            try:
                if viewer.is_closed:
                    break
                print("Waiting for a connection...")
                connection, client_address = server_socket.accept()
            except TimeoutError:
                continue
            
            print("Connection from", client_address)
            data = connection.recv(4) 
            if data:
                data_size = struct.unpack('!I', data)[0]
                # print(data_size)

                response_data = b''
                while len(response_data) < data_size:
                    packet = connection.recv(data_size - len(response_data))
                    if not packet:
                        break
                    response_data += packet

                response_data = pickle.loads(response_data)
                # print("Received response:", response_data)

                if "points" in response_data.keys():
                    viewer.set_points(response_data["points"])
                if "target_points" in response_data.keys():
                    viewer.set_target_points(response_data["target_points"])
                if "key_frames" in response_data.keys():
                    viewer.set_key_frames(response_data["key_frames"])
                if "curr_frame" in response_data.keys():
                    viewer.set_curr_frame(response_data["curr_frame"])

            else:
                connection.close()
                print("Lost connection, Waiting new connection")
                continue

    finally:
        print("Socket close")
        server_socket.close()



    # model_thread.join()

if __name__ == "__main__":
    import time

    K = np.array([
            7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 
            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]
        ).reshape((3,3)).astype(np.float64)

    WIDTH = 1241
    HEIGHT = 376

    # queue = Queue()
    # viewer_proc = Process(target=viewer_process, args=(queue, K, WIDTH, HEIGHT))
    # viewer_proc.start()


    viewer = Viewer3D()
    map_model = Map_model()
    map_model.points = [[1,1,1]]
    while 1:
        print("main")
        viewer.draw_map(map_model)
        time.sleep(5)


import pypangolin as pango
from OpenGL.GL import *

import threading
import multiprocessing
import socket
import struct
import pickle

import numpy as np
import numpy.linalg as la

class Viewer:
    def __new__(cls, *args):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Viewer, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, K, WIDTH, HEIGHT):
        pango.CreateWindowAndBind("Viewer", 1024, 768)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.s_cam = pango.OpenGlRenderState(
            pango.ProjectionMatrix(1024, 768, 2000, 2000, 512, 389, 0.1, 1000),
            pango.ModelViewLookAt(0, -100, -0.1, 0, 0, 0, 0.0, -1.0, 0.0)
        )

        self.handler = pango.Handler3D(self.s_cam)
        self.d_cam = (
            pango.CreateDisplay()
                .SetBounds(
                    pango.Attach(0.0), 
                    pango.Attach(1.0), 
                    pango.Attach.Pix(175), 
                    pango.Attach(1.0), 
                    -1024.0 / 768.0)
                .SetHandler(self.handler)
        )
        self.is_closed = False

        self.K = K
        self.Kinv = la.inv(self.K)
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

        self.points = None
        self.target_points = None
        self.key_frames = None
        self.curr_frame = None

    def set_points(self, points):
        self.points = points
        return True
    
    def set_target_points(self, points):
        self.target_points = points
        return True
    

    def set_key_frames(self, key_frames):
        self.key_frames = key_frames
        return True

    def set_curr_frame(self, curr_frame):
        self.curr_frame = curr_frame
        return True

    def run(self):
        glPointSize(10)
        while not pango.ShouldQuit():
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            pango.glDrawAxis(1)

            if self.points is not None:
                glColor4f(1.0, 1.0, 1.0, 1.0)
                pango.glDrawPoints(self.points)
            if self.target_points is not None:
                glColor4f(1.0, 0.0, 0.0, 1.0)
                pango.glDrawPoints(self.target_points)
            if self.key_frames is not None:
                glColor4f(1.0, 1.0, 1.0, 1.0)
                for key_frame in self.key_frames:
                    pango.glDrawFrustum(self.Kinv, self.WIDTH, self.HEIGHT, key_frame, 1.)
            if self.curr_frame is not None:
                glColor4f(0.0, 1.0, 0.0, 1.0)
                pango.glDrawFrustum(self.Kinv, self.WIDTH, self.HEIGHT, self.curr_frame, 1.)
            
            self.d_cam.Activate(self.s_cam)
            pango.FinishFrame()
        
        self.is_closed = True


def model(viewer):
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
                print(data_size)

                response_data = b''
                while len(response_data) < data_size:
                    packet = connection.recv(data_size - len(response_data))
                    if not packet:
                        break
                    response_data += packet

                response_data = pickle.loads(response_data)
                print("Received response:", response_data)

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

def viewer_process(K, WIDTH, HEIGHT):
    viewer = Viewer(K, WIDTH, HEIGHT)
    model_thread = threading.Thread(target=model, args=(viewer, ))
    model_thread.daemon = True
    model_thread.start()
    viewer.run()

    model_thread.join()

if __name__ == "__main__":
    K = np.eye(3, dtype=np.float64)
    WIDTH = 1
    HEIGHT = 1

    viewer_proc = multiprocessing.Process(target=viewer_process, args=(K, WIDTH, HEIGHT))
    viewer_proc.start()
    viewer_proc.join()
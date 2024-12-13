import yaml
import numpy as np
import numpy.linalg as la
from abc import ABC, abstractmethod
from enum import Enum

class Sensor_type(Enum):
    Monocular = 0
    Stereo = 1

class State(Enum):
    INIT_YET = 0
    INIT = 1
    CLOSED_IMG = 10
    INVALID_IMG = 20

class Settings(ABC):
    def __init__(self, dataset):
        with open("settings.yaml") as file:
            self.settings = yaml.safe_load(file)
        self._dataset = dataset
        self._sensor_type = self.get_value("sensor_type")
        self._width = self.get_value("width")
        self._height = self.get_value("height")

        self._img_path = self.get_value("img_path")
        self._imgr_path = self.get_value("imgr_path")
        self._calibration_path = self.get_value("calibration_path")
        self._timestamp_path = self.get_value("timestamp_path")

        self._kChi2Mono = self.get_value("kChi2Mono")
        self._kChi2Stereo = self.get_value("kChi2Stereo")
        self._cos_max_parallax = self.get_value("cos_max_parallax")
        self._scale_consistency_factor = self.get_value("scale_consistency_factor")
        self._scale_factor = self.get_value("scale_factor")
        self._desired_median_depth = self.get_value("desired_median_depth")

        self.make_K()
        self.make_timestamp()

    def get_value(self, key:str):
        try:
            return self.settings[self.dataset][key]
        except KeyError:
            return None
    
    @abstractmethod
    def make_K(self):
        pass
    @abstractmethod
    def make_timestamp(self):
        pass
        
    @property
    def dataset(self):
        return self._dataset
    @property
    def sensor_type(self):
        return self._sensor_type
    @property
    def width(self):
        return self._width
    @property
    def height(self):
        return self._height
    @property
    def img_path(self):
        return self._img_path
    @property
    def imgr_path(self):
        return self._imgr_path
    @property
    def timestamp(self):
        return self._timestamp_path
    @property
    def timestamp(self):
        return self._timestamp
    @property
    def calibration_path(self):
        return self._timestamp_path
    @property
    def K(self):
        return self._K
    @property
    def Rt(self):
        return self._Rt
    @property
    def kChi2Mono(self):
        return self._kChi2Mono
    @property
    def kChi2Stereo(self):
        return self._kChi2Stereo

    @property
    def cos_max_parallax(self):
        return self._cos_max_parallax
    @property
    def scale_consistency_factor(self):
        return self._scale_consistency_factor
    @property
    def scale_factor(self):
        return self._scale_factor
    @property
    def desired_median_depth(self):
        return self._desired_median_depth

class KITTI_Monocular_dataset(Settings):
    def __init__(self):
        super().__init__("KITTI_Monocular")

    def make_K(self):
        with open(self._calibration_path, "r") as file:
            Ps = file.readlines()
        # intrinsic parameter
        self._K = np.asarray(Ps[0].split(" ")[1:], dtype=np.float64).reshape(3,4)[:3,:3]
        P2 = np.asarray(Ps[1].split(" ")[1:], dtype=np.float64).reshape(3,4)
        # Rt is extrinsic parameter, 
        # P = K @ [R|t], [R|t] = K^-1 @ P
        self._Rt = la.inv(self._K)@P2

    def make_timestamp(self):
        with open(self._timestamp_path, "r") as file:
            self._timestamp = np.array([time for time in file.readlines()], dtype=np.float64)

if __name__ == "__main__":
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    
    settings = KITTI_Monocular_dataset()
    print(settings.sensor_type)
    print(settings.width, settings.height)
    print(settings.K)
    print(settings.Rt)
    print(settings.img_path)
    print(settings.imgr_path)
    print(settings.timestamp.shape)
    print(settings.kChi2Mono)
    print(settings.kChi2Stereo)
    print(settings.cos_max_parallax)


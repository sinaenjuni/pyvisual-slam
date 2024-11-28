import yaml
import numpy as np
from abc import ABC, abstractmethod

class Setting(ABC):
    @abstractmethod
    def get_img_dirs(self):
        pass
    @abstractmethod
    def get_img_dir(self):
        pass
    @abstractmethod
    def get_imgr_dir(self):
        pass
    @abstractmethod
    def get_calibration(self):
        pass

class KITTI_dataset(Setting):
    def __init__(self):
        with open("settings.yaml") as f:
            settings = yaml.safe_load(f)
        
        data_set = "KITTI"
        data_type = "PATH"
        self._img_dirs_path = settings[data_set][data_type]["img_dirs"]
        self._calibration_path = settings[data_set][data_type]["calibration"]
        self._timestamp_path = settings[data_set][data_type]["timestamp"]

        if len(self._img_dirs_path) > 1:
            self._img_dir_path = self._img_dirs_path[0]
            self._imgr_dir_path = self._img_dirs_path[1]
        
        with open(self._calibration_path, "r") as f:
            self._K = f.readlines()[0].split(" ")[1:]
            self._K = np.asarray(self._K, dtype=np.float64).reshape(3,4)[:3,:3]

        with open(self._timestamp_path, "r") as f:
            self._times = np.array([time for time in f.readlines()], dtype=np.float64)


    def get_img_dirs(self):
        return self._img_dirs_path
    
    def get_img_dir(self):
        return self._img_dir_path
    
    def get_imgr_dir(self):
        return self._imgr_dir_path
    
    def get_timestamp(self):
        return self._times
    
    def get_calibration(self):        
        return self._K

if __name__ == "__main__":
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    
    settings = KITTI_dataset()
    K = settings.get_calibration()
    img_path = settings.get_img_dir()
    imgr_path = settings.get_imgr_dir()
    timestamp = settings.get_timestamp()
    print(K)
    print(img_path)
    print(imgr_path)
    print(timestamp)
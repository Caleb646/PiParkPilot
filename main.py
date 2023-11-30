import numpy as np
import cv2 as cv
import math
import time
import asyncio
import logging
import logging.config
#from scipy.optimize import minimize
from typing import Tuple, List, Union, Dict

from pi_park.api import car, utils, vo


if __name__ == "__main__":
    car_ = car.Car(
        config_path="./pi_park/configs/pi_config.yml", 
        #img_log_dir="./logs/img_log/",
        should_camera_check=False)
    car_.drive_without_server_(target_fps=1)
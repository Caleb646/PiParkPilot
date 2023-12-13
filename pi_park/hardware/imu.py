import time
import board
import busio
import serial
import adafruit_bno055

import numpy as np
import multiprocessing as mp
from typing import Optional, List, Dict

class Mode:
    CONFIG_MODE = 0x00
    ACCONLY_MODE = 0x01
    MAGONLY_MODE = 0x02
    GYRONLY_MODE = 0x03
    ACCMAG_MODE = 0x04
    ACCGYRO_MODE = 0x05
    MAGGYRO_MODE = 0x06
    AMG_MODE = 0x07
    IMUPLUS_MODE = 0x08 
    """
    Fusion Mode with Accel and Gyro. `Orientation is relative` to initial starting position.
    """
    COMPASS_MODE = 0x09
    M4G_MODE = 0x0A
    NDOF_FMC_OFF_MODE = 0x0B
    NDOF_MODE = 0x0C
    """
    Fusion Mode with Accel, Magnometer and Gyro. `Absolute orientation` = orientation of the sensor with respect to
    the earth and its magnetic field.
    """

def get_time():
    return time.perf_counter()

class IMU:
    def __init__(self, hertz=25, x_accel_off=0, y_accel_off=0.1):
        i2c = board.I2C()
        self.x_accel_off = x_accel_off
        self.y_accel_off = y_accel_off
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)
        # Set calibration values
        self.sensor.offsets_magnetometer = (-97, -73, -463)
        self.sensor.offsets_gyroscope = (0, -2, 4)
        self.sensor.offsets_accelerometer = (-2, -57, -17)
        # Only using gyro and accel
        self.sensor.mode = Mode.IMUPLUS_MODE

        self.imu2cam = np.asarray([
            # left (pos x) and right (neg x) are the same in both imu and cam 
            [1, 0, 0], 
            # IMU: up is pos z and down is neg z ---- Cam: up is pos y and down is neg y
            [0, 0, 1],
            # IMU: forward is neg y and backward is pos y ---- Cam: forward is pos z and backward is neg z
            [0, -1, 0],
        ])
        self.last_update = get_time()
        self.max_hertz = hertz
        # store 1 second worth of readings
        self.max_stored_readings = self.max_hertz #* 2
        self.stored_readings = np.zeros((self.max_stored_readings, 7), dtype=np.float32)
        self.cur_readings_idx = 0
        """
        Store: (reading timestamp, ax, ay, az, gx, gy, gz)
        """

    def read_(self):
        return (
            self.sensor.acceleration, 
            self.sensor.euler 
        )

    def transform_reading(self, reading: np.ndarray):
        assert reading.shape[0] == 7, "Invalid reading shape"
        transformed_reading = np.zeros((7,), dtype=np.float32)
        accel = np.dot(reading[1:4], self.imu2cam)
        euler = np.dot(reading[4:], self.imu2cam)
        transformed_reading[0] = reading[0]
        transformed_reading[1:4] = accel
        transformed_reading[4:] = euler
        return transformed_reading

    def get_readings(self, timestamp: Optional[float]=None):
        if timestamp is None:
            return np.asarray([self.transform_reading(
                self.stored_readings[max(self.cur_readings_idx-1, 0)]
            )])
        
        selected_readings = ((self.stored_readings[:, 0] - timestamp) > 0).astype(bool) 
        selected_readings = self.stored_readings[selected_readings]
        for i in range(selected_readings.shape[0]):
            selected_readings[i] = self.transform_reading(selected_readings[i])
        return selected_readings

    def step(self):
        start = get_time()
        accel, euler = self.read_()
        self.stored_readings[self.cur_readings_idx] = [
            get_time(), 
            accel[0] - self.x_accel_off, accel[1] - self.y_accel_off, accel[2], 
            euler[0], euler[1], euler[2]
        ]
        self.cur_readings_idx = (self.cur_readings_idx + 1) % self.max_stored_readings
        return get_time() - start

    def run_(self, queue: Optional[mp.Queue]):
        while True:
            step_time = self.step()
            sleep_time = max((1 / self.max_hertz) - step_time, 0.01)
            if queue:
                queue.put(self.get_readings(get_time() - 1).tolist())
            if sleep_time > 0:
                time.sleep(sleep_time)

def print_run(queue: mp.Queue):
    while True:
        print(queue.get())
        time.sleep(0.5)

def parallel_run():
    queue = mp.Queue()
    imu = IMU(hertz=2)
    process1 = mp.Process(target=imu.run_, args=(queue,))
    process2 = mp.Process(target=print_run, args=(queue,))

    process1.start()
    process2.start()

    process1.join()
    process2.join()

if __name__ == "__main__":
    parallel_run()
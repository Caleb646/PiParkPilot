import time
import board
import busio
import serial
import threading
import adafruit_bno055

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

i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)
sensor.mode = Mode.IMUPLUS_MODE

while True:
    print(sensor.linear_acceleration)
    time.sleep(0.25)
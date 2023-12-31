import numpy as np
import cv2 as cv
import math
import time
import asyncio
import logging
import sys
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import multiprocessing as mp
from typing import Tuple, List, Union, Dict, Optional
import os

import pi_park.visual_odometry as vo
from pi_park.stereo_camera import StereoCamera
import pi_park.server as ws_server
import pi_park.utils as utils
from pi_park.hardware.imu import IMU

# NOTE: If car.py is imported multiple times this can cause the logger
# to be instantiated multiple time. 
logger = logging.getLogger(__name__)

def generate_path(
        start_pos: Tuple[float, float, float], 
        end_pos: Tuple[float, float, float], 
        final_time=60,
        normalize=True
        ) -> np.ndarray:
    
    def to_p_domain(t, tf):
        return (10/tf**3) * t**3 - (15/tf**4) * t**4 + (6/tf**5) * t**5
    
    def minimize_func(p, x):
        x_der = (x[2] + 2 * x[3] * p + 3 * x[4] * p**2)**2
        y_der = (x[5] + 2 * x[6] * p + 3 * x[7] * p**2)**2
        return x_der**2 + y_der**2
    
    start_x, start_y, start_theta = start_pos
    end_x, end_y, end_theta = end_pos

    if not (abs(end_x) > 0 and abs(end_y) > 0):
        logger.error(f"end_x -> {end_x} and end_y -> {end_y} can NOT be 0.")
        raise ValueError(f"end_x -> {end_x} and end_y -> {end_y} can NOT be 0.")
    if not (start_theta == 0 and end_theta == 0):
        logger.error(f"starting -> {start_theta} and ending -> {end_theta} thetas must both be 0.")
        raise ValueError(f"starting -> {start_theta} and ending -> {end_theta} thetas must both be 0.")
    # initial_parameters = [1, 1, 1.3, 0.7, 0.8, 1.9, 1.9, 1.2]
    # constraints = (
    #         {'type': 'eq', 'fun': lambda x: x[0] - start_x},
    #         {'type': 'eq', 'fun': lambda x: x[4] - start_y},
    #         {'type': 'eq', 'fun': lambda x: x[5] - math.tan(math.radians(start_theta))},
    #         {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - end_x},
    #         {'type': 'eq', 'fun': lambda x: x[4] + x[5] + x[6] + x[7] - end_y},
    #         {'type': 'eq', 'fun': lambda x: x[5] + 2 * x[6] + 3 * x[7] - math.tan(math.radians(end_theta))}
    #     )
    # B = minimize(
    #     lambda x : minimize_func(0, x), initial_parameters, tol=1e-6, constraints=constraints
    #     ).x
    # points = [None for _ in range(final_time)]
    # for i in range(final_time):
    #     p_out = to_p_domain(i, tf=final_time)
    #     x = B[0] + B[1] * p_out + B[2] * p_out**2 + B[3] * p_out**3
    #     y = B[4] + B[5] * p_out + B[6] * p_out**2 + B[7] * p_out**3
    #     points[i] = [x, y]
    # points = np.array(points)

    # Try generating the cubic spline with control points to enforce the starting 
    # and ending positions' thetas
    cs = CubicSpline(x=[start_x, start_x + 0.05, end_x - 0.05, end_x], y=[start_y, start_y, end_y, end_y])
    # Controls how many points will be returned
    xs = np.arange(0, end_x, 0.2)
    ys = cs(xs)
    points = np.hstack([xs[:, np.newaxis], ys[:, np.newaxis]])
    
    if normalize:
        points[:, 0] /= end_x
        points[:, 1] /= end_y
    return points

def detect_lines(left_img: cv.Mat):
    gray = np.uint8(left_img)
    edges = cv.Canny(gray, 150, 255, apertureSize = 3)
    # lines = cv.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
    """
    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    lines: A vector that will store the parameters (r,θ) of the detected lines
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    """
    lines = cv.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50, lines=None, minLineLength=50, maxLineGap=10
        )
    """
    dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    lines: A vector that will store the parameters (xstart,ystart,xend,yend) of the detected lines
    rho : The resolution of the parameter r in pixels. We use 1 pixel.
    theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    threshold: The minimum number of intersections to "*detect*" a line
    minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    maxLineGap: The maximum gap between two points to be considered in the same line.
    """
    return np.squeeze(lines)

def lines_intersections(left_img: cv.Mat):
    lines = detect_lines(left_img)
    def determinate(cline, nline):
        x1, y1, x2, y2 = cline
        x3, y3, x4, y4 = nline
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 0.1:
            return None, None
        x = (x1 * y1 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * x4 - y3 * x4)
        y = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        return x / denominator, y / denominator

    intersections = []
    for i in range(lines.shape[0]):
        for j in range(i + 1, lines.shape[0]):
            cline = lines[i]
            nline = lines[j]
            x, y = determinate(cline, nline)
            if x is not None and y is not None:
                intersections.append([x, y])
    return np.array(intersections)

def detect_corners(left_img: cv.Mat):
    corners = cv.cornerHarris(np.float32(left_img), 2, 3, 0.04)
    return corners > 0.01 * corners.max()


class Car:
    def __init__(
            self, 
            config_path: str,
            ip="127.0.1.1",
            img_log_dir: Optional[str] = None,
            should_camera_check = False,
            imu_mp_queue: Optional[mp.Queue] = None,
            max_speed_mph = 1.0
            ):
        #self.logger = logging.getLogger(__name__)
        self.logger = logger
        self.config = utils.load_yaml_file(config_path)
        self.cam = StereoCamera(
            self.config["left_cam_id"], self.config["right_cam_id"], self.config["stereo_map_path"]
            )
        # TODO: if forward is in the direction the camera is initially facing
        # and the camera's initial facing is not parallel with the side of the car 
        # then the car's forward/backward motion will have to be offset somehow.
        # The angle between the camera's initial facing direction and side of the car
        # will have to be a known value.
        self.mat_current_position = np.eye(4)[:3, :]
        """
        ``shape -> (3, 4). units -> meters for translation column``. 
        Stores the current position of the left camera in world space. 
        x will be left and right with left being negative and left being positive. 
        y should be remain relative constant as the car shouldn't be moving up and down. 
        z will be forward and backward with forward being the direction that the camera is initially facing.
        """
        self.visual_od = vo.VisualOdometry(self.cam.left_proj, self.cam.right_proj)

        self.left_prev_img: Union[cv.Mat, None] = None
        self.right_prev_img: Union[cv.Mat, None] = None

        self.current_path: Union[np.ndarray, None] = None 
        self.end_pos: Union[List[float, float, float], None] = None
        self.aspect_ratio: Union[float, None] = None
        """
        units -> meters. Contains the x, z, and y axis rotation values for the
        desired end position.
        """
        self.car_max_steer_angle = self.config["max_steering_angle"]
        """
        The maximum steering angle of the car specified in degrees.
        """
        self.axles_distance = self.config["axles_distance"]
        """
        The distance between the front and rear axles in meters.
        """
        self.server_ = ws_server.WebSocketServer(host=ip)
        self.calibrated_frame_size = tuple(self.config["frame_size"])

        if should_camera_check:
            self.visually_check_cameras()

        self.img_log_dir = img_log_dir
        if self.img_log_dir is not None:
            utils.clear_dir(self.img_log_dir)

        self.imu_mp_queue = imu_mp_queue
        self.max_speed_ms = max_speed_mph / 2.237
        self.prev_step_start_time = utils.get_time()
        self.cur_step_start_time = utils.get_time()
        #self.imu = None
        #if self.imu_mp_queue is None:
            #from pi_park.hardware.imu import IMU
            #self.imu = IMU(hertz=25, max_speed_ms=self.max_speed_ms)

    @property
    def position(self):
        y = self.mat_current_position[1, 3]
        return (self.xpos, y, self.zpos, self.yrot)
    
    @property
    def xpos(self):
        return self.mat_current_position[0, 3]
    
    @property
    def zpos(self):
        return self.mat_current_position[2, 3]
    
    @property
    def yrot(self):
        """
        Returns
        -----------
        ``units -> degrees.`` The rotation around the y axis.
        """
        return utils.decompose_rotation_matrix(self.mat_current_position[:3, :3])[1]
    
    @property
    def xpos_normed(self):
        return self.xpos / (self.end_pos[0] - self.xpos)
    
    @property
    def zpos_normed(self):
        return self.zpos / (self.end_pos[1] - self.zpos)
    
    def visually_check_cameras(self):
        while self.cam.is_opened():
            left_img, right_img = self.cam.read()
            k = cv.waitKey(5)
            if k == ord("q") or k == 27: # press q or esc to quit
                break
            cv.imshow("Should be Left Camera", left_img)
            cv.imshow("Should be Right Camera", right_img)
        cv.destroyAllWindows()

    def process_imu_readings(
        self, readings: np.ndarray
        ):
        #print(f"IMU has [{readings.shape[0]}]")
        #print(self.prev_step_start_time, self.cur_step_start_time, np.squeeze(readings[:, 0]))

        selected_readings = IMU.get_readings_between(
            readings, self.prev_step_start_time, self.cur_step_start_time
            )
        timestamps = np.argsort(selected_readings[:, 0])
        selected_readings = selected_readings[timestamps]
        num_readings = selected_readings.shape[0]
        print(f"Selected IMU [{num_readings}] readings")
        print(f"Selected IMU readings Acceleration: [{np.round(np.squeeze(selected_readings[:, 1:4]), decimals=3)}]")
        total_dt = self.cur_step_start_time - self.prev_step_start_time
        if num_readings == 0:
            return None

        def integrate(
            a: Tuple[float, float, float], b: Tuple[float, float, float], dt: float
            ):
            ax1, ay1, az1 = a[0], a[1], a[2]
            ax2, ay2, az2 = b[0], b[1], b[2]
            vx, vy, vz = ((ax1+ax2)/2)*dt, ((ay1+ay2)/2)*dt, ((az1+az2)/2)*dt  
            return np.asarray([vx, vy, vz])

        if num_readings == 1:
            _, accel, euler = selected_readings[0, 0], selected_readings[0, 1:4], selected_readings[0, 4:]
            # NOTE: the euler angles given by the sensor is not the 
            # rotation matrix from sensor position a to b.
            # It is the sensor's rotational position
            rmat = utils.euler2rotation(euler[0], euler[1], euler[2])
            tvec = integrate(accel, total_dt)
            final_position = vo.project_position(self.mat_current_position, np.eye(3, 3), tvec)
            final_position[:3, :3] = rmat
            return final_position

        
        vels = np.zeros((num_readings - 1, 4))
        for i in range(num_readings - 1):
            prev_t, p_accel = selected_readings[i, 0], selected_readings[i, 1:4]
            cur_t, c_accel = selected_readings[i+1, 0], selected_readings[i+1, 1:4]
            dt = cur_t - prev_t
            vels[i] = np.hstack([[cur_t], integrate(p_accel, c_accel, dt)])

        tvec = np.zeros((3,))
        for i in range(vels.shape[0] - 1):
            prev_t, p_vel = vels[i, 0], vels[i, 1:]
            cur_t, c_vel = vels[i+1, 0], vels[i+1, 1:]
            dt = cur_t - prev_t
            tvec += integrate(p_vel, c_vel, dt)

        euler = selected_readings[-1, 4:]
        rmat = utils.euler2rotation(euler[0], euler[1], euler[2])
        final_position = vo.project_position(self.mat_current_position, np.eye(3, 3), tvec)
        final_position[:3, :3] = rmat
        return final_position

    def process_cam_readings(
        self,
        cam_new_rmat: np.ndarray, 
        cam_new_tvec: np.ndarray
        ):
        trans = np.squeeze(cam_new_tvec)
        
        dt = self.cur_step_start_time - self.prev_step_start_time
        px, py, pz, pyr = self.position

        cxr, cyr, czy = utils.decompose_rotation_matrix(cam_new_rmat)
        cx, cy, cz = trans[0], trans[1], trans[2]

        dx = (cx - px) / dt
        dz = (cz - pz) / dt 
        speed = np.sqrt((dx**2) + (dz**2))

        #print(f"Camera Speed: {(dx, dz, speed)}")
        if speed > self.max_speed_ms:
            zp = dz / speed
            xp = dx / speed

            new_dx = self.max_speed_ms * xp
            new_dz = self.max_speed_ms * zp

            new_x = new_dx * dt #+ px
            new_z = new_dz * dt #+ pz

            #print("Adjusting X", xp, new_dx, new_x)
            #print("Adjusting Z", zp, new_dz, new_z)

            trans[0] = new_x
            trans[2] = new_z

        position_guess = vo.project_position(
                self.mat_current_position, cam_new_rmat, trans
                )
        return position_guess

    def compare_cam_imu_position_guesses(
        self, 
        cam_position_guess: np.ndarray, 
        imu_position_guess: np.ndarray
        ):
        if cam_position_guess is None and imu_position_guess is not None:
            return imu_position_guess
        if cam_position_guess is not None and imu_position_guess is None:
            return cam_position_guess
        if cam_position_guess is None and imu_position_guess is None:
            return None

        cam_rmat = cam_position_guess[:3, :3]
        cam_tvec = cam_position_guess[:, 3]
        imu_rmat = imu_position_guess[:3, :3]
        imu_tvec = imu_position_guess[:, 3]

        translation_error = np.sqrt(np.sum((imu_tvec - cam_tvec)**2))
        rotation_error = np.sqrt(np.sum((cam_rmat - cam_tvec)**2))
        print(f"IMU - Cam Translation RMSE: {np.sqrt(np.sum((imu_tvec - cam_tvec)**2))}")
        # NOTE: The IMU is better with rotation
        # so uses its rotation guess over the cameras. 
        # The IMU's rotation is the sensor's rotational position not a transformation
        # to the sensor's rotational position
        rmat = np.eye(3, 3)
        # TODO: decide how to mesh these two together
        tvec = (cam_tvec + imu_tvec) / 2
        #tvec = imu_tvec
        position = vo.project_position(self.mat_current_position, rmat, tvec)
        position[:3, :3] = imu_rmat
        return position

    def get_imu_readings_with_mp_queue(self):
        if self.imu_mp_queue is not None:
            # NOTE: MP Queue is a FIFO queue
            # So the last value is the most up-to-date.
            # So iterate until a mp.queue.Empty exception is thrown
            # and return the most recent valid value
            r, max_iter = None, 5
            try:           
                for i in range(max_iter):
                    temp = self.imu_mp_queue.get(block=False)
                    if temp is not None:
                        r = temp
                if r is not None:
                    return np.asarray(r)
            except mp.queues.Empty:
                if r is not None:
                    return np.asarray(r)
                self.logger.warn("Failed to get IMU readings from MP Queue because it is EMPTY")
                return None
        return None

    def get_imu_readings(self):
        if self.imu_mp_queue is not None:
            return self.get_imu_readings_with_mp_queue()
        return self.imu.get_most_recent_readings(50)

    def sample_imu_readings(self, num_readings=1):
        if self.imu is not None:
            self.imu.step(step_count=num_readings)
            return self.imu.get_most_recent_readings(num_readings)
        return None

    def step(self):
        start_time = utils.get_time()
        self.cur_step_start_time = start_time
        # Sample from the IMU but don't use the readings yet
        # self.sample_imu_readings(num_readings=2)
        left_next_img, right_next_img = self.cam.read()
        # Possible States -> We've failed to get the next set of images
        if left_next_img is None or right_next_img is None:
            self.left_prev_img, self.right_prev_img = None, None
            self.logger.error("Failed to get next image frames.")
            return utils.get_time() - start_time
        # Possible States -> Its the first frame or something failed and we've reset
        if self.left_prev_img is None or self.right_prev_img is None:
            # TODO: probably want to find the end destination here.
            self.end_pos = (3, 3, 0) # x, z, y rotation

            # Set the max allowed depth to the total distance from the camera's starting position
            # to the ending position plus 2 
            dist = np.squeeze(np.sqrt((self.xpos - self.end_pos[0])**2 + (self.zpos - self.end_pos[1])**2))
            #self.visual_od.max_depth_m = max(dist + 2, 7)
            # TODO: Change
            self.visual_od.max_depth_m = 10

            self.aspect_ratio = (self.end_pos[1] - self.zpos) / (self.end_pos[0] - self.xpos)
            x_car, _, z_car, y_rot = self.position

            # Only create the path once
            if self.current_path is None:
                self.current_path = generate_path(
                    (x_car, z_car, y_rot),
                    self.end_pos
                    )
            self.left_prev_img, self.right_prev_img = left_next_img, right_next_img
            self.logger.warn("Previous images are none")
            return utils.get_time() - start_time

        motion_res: vo.MotionEstimateResult = self.visual_od.estimate_motion(
            self.left_prev_img, 
            self.right_prev_img, 
            left_next_img, 
            right_next_img,
            use_kalman_filter=False
            )
        if motion_res.estimate_successful:
            rmat, tvec =  motion_res.rmat, motion_res.tvec
            imu_readings = self.get_imu_readings() #self.imu.get_readings_between(self.prev_step_start_time, self.cur_step_start_time)
            imu_position_guess = self.process_imu_readings(imu_readings)
            if imu_position_guess is not None:
                print(f"IMU Position Guess: {np.squeeze(imu_position_guess[:, 3])}")
            # Try and adjust the new camera readings using self.max_max_ms
            # because the car should never exceed 1-2 mph. So the x and z values of the translation 
            # can be rescaled
            cam_position_guess = self.process_cam_readings(rmat, tvec)
            # Next use the IMU data to adjust the rotation of the camera and maybe the translation
            # of it as well. 
            new_position = self.compare_cam_imu_position_guesses(cam_position_guess, imu_position_guess)
            # Take the updated camera matrix (extrinsic + intrinsic) and find the inverse.
            # Taking the matrix product of the current position with the inverse of the new camera 
            # matrix moves the current position to the new world coordinate.
            # The inverse of the new camera matrix gives creates a transformation from camera space to
            # world space instead of vice versa.
            if new_position is not None:
                self.mat_current_position = new_position
                self.left_prev_img, self.right_prev_img = left_next_img, right_next_img
                self.logger.info(f"New Position: {self.position}")
            else:
                self.logger.info(f"Both the IMU and Camera positions guesses are None")
        else:
            self.logger.error("Failed to estimate motion")
            # TODO: Temporary
            self.left_prev_img, self.right_prev_img = None, None

        # Sample from the IMU again
        # self.sample_imu_readings(num_readings=2)
        self.prev_step_start_time = self.cur_step_start_time
        return utils.get_time() - start_time
    
    def at_end_pos(self, dist_from=0.3):
        if self.end_pos is None:
            self.logger.warn("self.end_pos is None")
            return False
        d = np.squeeze(np.sqrt((self.xpos - self.end_pos[0])**2 + (self.zpos - self.end_pos[1])**2))
        self.logger.info(f"{d} meters away from end destination")
        if d <= dist_from: # 0.3 meters = ~1 foot
            return True
        return False
    
    async def drive(self, target_fps=15, wait_on_connection=True) -> None:
        run_server = asyncio.create_task(self.server_.run())
        driving = asyncio.create_task(
            self.drive_(
                target_fps=target_fps, wait_on_connection=wait_on_connection
                )
            )
        done, pending = await asyncio.wait(
            [run_server, driving],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    
    async def drive_(self, target_fps=15, wait_on_connection=True):
        while not self.at_end_pos():
            if self.server_.has_connections() or not wait_on_connection:
                step_time = self.step()
                sleep_time = max((1/target_fps) - step_time, 0.01)
                self.logger.info(
                    f"Step Time: {step_time} seconds ---- Sleep Time {sleep_time} seconds"
                    )
                await self.server_.send(self.get_network_data())
                await asyncio.sleep(sleep_time)
            else:
                self.logger.info(
                    f"Server at: {f'{self.server_.host}:{self.server_.port}'} waiting for connection."
                    )
                await asyncio.sleep(0.5)
            
    def drive_without_server_(self, target_fps=15):
        while not self.at_end_pos():
            step_time = self.step()
            sleep_time = max((1/target_fps) - step_time, 0.01)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.logger.info(f"Step Time: {step_time} seconds ---- Sleep Time {sleep_time} seconds")

    async def shutdown(self):
        self.cam.shutdown()
        self.left_prev_img, self.right_prev_img = None, None
        await self.server_.shutdown()

    def get_network_data(self):
        return {
            "aspect_ratio": self.aspect_ratio, # (z_end - z_start) / (x_end - x_start) or height / width
            "cur_pos": (self.xpos_normed, self.zpos_normed, self.yrot), # (x, z, y_rotation)
            "cur_path": self.current_path.tolist()
        }
        
if __name__ == "__main__":
    #car = Car()
    #car.drive_without_server_(target_fps=10, verbose=True)
    from copy import deepcopy
    img = cv.imread("./data/test_data/end_point_detection/parking_spot.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    thresh = deepcopy(img)
    thresh[( np.sum(thresh, axis=2) / 3 ) < 250] = [0, 0, 0]
    thresh = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)
    thresh = cv.Canny(thresh, 150, 255, apertureSize = 3)
    #kernel = np.ones((3, 3), np.uint8) 
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # #
    # thresh = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel, iterations=1)
    # thresh = cv.erode(thresh, kernel, iterations=1)

    intersections = lines_intersections(thresh)
    print(intersections.shape)

    corners = detect_corners(thresh)
    print(corners.shape, np.sum(corners))
    img[corners == 1] = [0, 0, 255]

    
    #assert False
    #cv.imshow('dst', img)
    cv.imshow('dst', thresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ws = ws_server.WebSocketServer(host="localhost")
    path = generate_path((0, 0, 0), (5.31114, 1.8288, 0))
    async def drive_():
        while True:
            if ws.connections_:
                for point in path:
                    await ws.send({
                        "cur_pos": point.tolist() + [0.1],
                        #"cur_path": path.tolist()
                        })
                    await asyncio.sleep(0.3)
                break
            await asyncio.sleep(0.1)

    async def drive() -> None:
        run_server = asyncio.create_task(ws.run())
        driving = asyncio.create_task(drive_())
        done, pending = await asyncio.wait(
            [run_server, driving],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

    asyncio.run(drive())


import numpy as np
import cv2 as cv
import math
import time
import asyncio
from scipy.optimize import minimize
from typing import Tuple, List, Union, Dict

import visual_odometry as vo
import server

def generate_path(
        start_pos: Tuple[float, float, float], 
        end_pos: Tuple[float, float, float], 
        final_time=60
        ) -> List[float, float]:
    
    def to_p_domain(t, tf):
        return (10/tf**3) * t**3 - (15/tf**4) * t**4 + (6/tf**5) * t**5
    
    def minimize_func(p, x):
        x_der = (x[2] + 2 * x[3] * p + 3 * x[4] * p**2)**2
        y_der = (x[5] + 2 * x[6] * p + 3 * x[7] * p**2)**2
        return x_der**2 + y_der**2
    
    start_x, start_y, start_theta = start_pos
    end_x, end_y, end_theta = end_pos
    initial_parameters = [1, 1, 1.3, 0.7, 0.8, 1.9, 1.9, 1.2]
    constraints = (
            {'type': 'eq', 'fun': lambda x: x[0] - start_x},
            {'type': 'eq', 'fun': lambda x: x[4] - start_y},
            {'type': 'eq', 'fun': lambda x: x[5] - math.tan(math.radians(start_theta))},
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - end_x},
            {'type': 'eq', 'fun': lambda x: x[4] + x[5] + x[6] + x[7] - end_y},
            {'type': 'eq', 'fun': lambda x: x[5] + 2 * x[6] + 3 * x[7] - math.tan(math.radians(end_theta))}
        )
    B = minimize(
        lambda x : minimize_func(0, x), initial_parameters, tol=1e-6, constraints=constraints
        ).x
    points = [None for _ in range(final_time)]
    for i in range(final_time):
        p_out = to_p_domain(i, tf=final_time)
        x = B[0] + B[1] * p_out + B[2] * p_out**2 + B[3] * p_out**3
        y = B[4] + B[5] * p_out + B[6] * p_out**2 + B[7] * p_out**3
        points[i] = [x, y]
    return np.array(points)


class Car:
    # TODO: I need the car configuration as well such as maximum steering angle
    # and the distance between the front and rear axles.
    def __init__(
            self, 
            left_cam_proj_mat: np.ndarray, 
            right_cam_proj_mat: np.ndarray
            ):
        self.left_proj = left_cam_proj_mat
        """
        shape -> (3, 4).
        The left camera's initial extrinsic matrix will be the origin for the world.
        Every point's position will be relative to it.
        """
        self.right_proj = right_cam_proj_mat
        """
        shape -> (3, 4).
        """
        _, rotation, translation, euler_angles = vo.decompose_projection_matrix(
            self.left_proj, return_euler_angles=True
            )
        # TODO: if forward is in the direction the camera is initially facing
        # and the camera's initial facing is not parallel with the side of the car 
        # then the car's forward/backward motion will have to be offset somehow.
        # The angle between the camera's initial facing direction and side of the car
        # will have to be a known value.
        self.current_y_rotation = euler_angles[1]
        """
        ``units -> degrees.`` This is the rotation around the y axis.
        """
        #self.current_position = np.vstack([rotation, translation])
        self.current_position = translation # [x, y, z]
        """
        ``shape -> (3, 1). units -> meters``. Stores the current position of the left camera in world space. 
        x will be left and right with left being negative and left being positive. 
        y should be remain relative constant as the car shouldn't be moving up and down. 
        z will be forward and backward with forward being the direction that the camera is initially facing.
        """
        self.visual_od = vo.VisualOdometry(self.left_proj, self.right_proj)
        self.left_cap = cv.VideoCapture(0)
        self.right_cap = cv.VideoCapture(2)
        assert self.left_cap.isOpened() and self.right_cap.isOpened()

        self.left_prev_img: Union[cv.Mat, None] = None
        self.right_prev_img: Union[cv.Mat, None] = None

        self.current_path: np.ndarray = np.zeros((1,1))
        self.end_pos: Union[List[float, float, float], None] = None
        """
        units -> meters. Contains the x, z, and y axis rotation values for the
        desired end position.
        """
        self.car_max_steer_angle = 25 
        """
        The maximum steering angle of the car specified in degrees.
        """
        self.axles_distance = 2
        """
        The distance between the front and rear axles in meters.
        """
        self.server_ = server.WebSocketServer()

    @property
    def xpos(self):
        return self.current_position[0]
    
    @property
    def zpos(self):
        return self.current_position[2]

    def step(self, verbose=False):
        start_time = time.time()
        left_next_img, right_next_img = self.read_images(verbose=verbose)
        if self.left_prev_img is None or self.right_prev_img is None:
            # TODO: probably want to find the end destination here.
            self.end_pos = (-10, 10, self.current_y_rotation)
            self.current_path = generate_path(
                (self.xpos, self.zpos, self.current_y_rotation),
                self.end_pos
                ) # NOTE: Graphing this path will be weird unless its translated
            self.left_prev_img, self.right_prev_img = left_next_img, right_next_img
            return time.time() - start_time
        updated_rotation, updated_position, _, _ = self.visual_od.estimate_motion(
            self.left_prev_img, self.right_prev_img, left_next_img, right_next_img
            )
        _, y_theta, _ = vo.decompose_rotation_matrix(updated_rotation)
        # Is the updated path following the generated path? Do I even need to check this?
        self.current_position = updated_position
        self.current_y_rotation = y_theta
        self.left_prev_img, self.right_prev_img = left_next_img, right_next_img
        return time.time() - start_time
    
    def at_end_pos(self, dist_from=0.3, verbose=False):
        if self.end_pos is None:
            return False
        d = np.sqrt((self.xpos - self.end_pos[0])**2 + (self.zpos - self.end_pos[1]))
        if verbose:
            print(f"{round(d, 2)} meters away from end destination")
        if d <= dist_from: # 0.3 meters = ~1 foot
            return True
        return False
    
    async def drive(self, verbose=False) -> None:
        run_server = asyncio.create_task(self.server_.run())
        driving = asyncio.create_task(self.drive_(verbose=verbose))
        done, pending = await asyncio.wait(
            [run_server, driving],
            return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    
    async def drive_(self, target_fps=15, verbose=False):
        while not self.at_end_pos(verbose=True):
            step_time = self.step()
            sleep_time = max((1/target_fps) - step_time, 0.01)
            await self.server_.send(self.get_network_data())
            await asyncio.sleep(sleep_time)
            if verbose:
                print(f"Slept for {sleep_time} seconds")

    def get_network_data(self):
        return {
            "current_pos": self.current_position.tolist(),
            "current_y_axis_rotation": self.current_y_rotation,
            "current_path": self.current_path.tolist()
        }
        
    def read_images(self, to_gray_scale=True, verbose=False):
        if self.left_cap.isOpened() and self.right_cap.isOpened():
            # TODO: reading the left and right images may need to be synchronized 
            # better than this.
            s1 = time.time()
            left = self.left_cap.read()
            e1 = time.time()
            s2 = time.time()
            right = self.right_cap.read()
            e2 = time.time()
            if verbose:
                print(f"Left Image -> Duration: {e1 - s1} End: {e1} -- Right Image -> Duration: {e2 - s2} End: {e2}")
            if to_gray_scale:
                return  cv.cvtColor(left, cv.COLOR_BGR2GRAY), cv.cvtColor(right, cv.COLOR_BGR2GRAY)
            return left, right
        return None, None
    
if __name__ == "__main__":
    car = Car()
    car.drive(verbose=True)
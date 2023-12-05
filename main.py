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


# Times on Laptop:
# Depth Map Creation Time: 0.017440080642700195 seconds

if __name__ == "__main__":
    car_ = car.Car(
        config_path="./pi_park/configs/basic_config.yml", 
        #ip="192.168.1.38",
        #img_log_dir="./logs/img_log/",
        should_camera_check=False
        )
    try:
        #car_.drive_without_server_(target_fps=0.25)
        asyncio.run(car_.drive(target_fps=0.25, wait_on_connection=False))
    except:
        #car_.shutdown()
        asyncio.run(car_.shutdown())
        asyncio.get_event_loop().stop()

    # try:
    #     end_x, end_z = 5.3114, 1.8288
    #     ws = car.ws_server.WebSocketServer(host="192.168.1.38")
    #     path = car.generate_path((0, 0, 0), (end_x, end_z, 0))
    #     async def drive_():
    #         cur_x, cur_z = 0, 0
    #         while True:
    #             if ws.connections_:
    #                 for point in path:
    #                     cur_x, cur_z, y_rot = cur_x + 0.1, cur_z, cur_x + 0.1
    #                     await ws.send({
    #                         "cur_pos": (cur_x/end_x, cur_z/end_z, y_rot),
    #                         "cur_path": path.tolist()
    #                         })
    #                     await asyncio.sleep(1)
    #                 break
    #             await asyncio.sleep(0.1)

    #     async def drive() -> None:
    #         run_server = asyncio.create_task(ws.run())
    #         driving = asyncio.create_task(drive_())
    #         done, pending = await asyncio.wait(
    #             [run_server, driving],
    #             return_when=asyncio.FIRST_COMPLETED
    #         )
    #         for task in pending:
    #             task.cancel()

    #     asyncio.run(drive())
    # except:
    #     ws.shutdown()
    #     asyncio.get_event_loop().stop()
    # car_ = car.Car(
    #     config_path="./pi_park/configs/pi_config.yml", 
    #     #ip="192.168.1.38",
    #     #img_log_dir="./logs/img_log/",
    #     should_camera_check=False
    #     )
    
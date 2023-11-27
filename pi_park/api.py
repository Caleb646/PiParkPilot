import logging
import logging.config
import os

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs\\logger.conf")
assert os.path.exists(log_file_path), f"Log Config path: {log_file_path} does NOT exist"
logging.config.fileConfig(log_file_path, disable_existing_loggers=False)

import pi_park.utils as utils
import pi_park.visual_odometry as vo
import pi_park.car as car
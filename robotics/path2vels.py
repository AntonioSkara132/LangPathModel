import numpy as np
import matplotlib.pyplot as plt
from src.nn import LangPathModel

def calc_point_vels(path: np.ndarray, target_velocity=1.0, ramp_up_time=100.0, ramp_down_time=140.0):
	""" 
	function that takes paths and calculates velocities according to the wanted trapesoid side profile
	returns: ndarray od size [Sequence length, 2], [velocity_x, velocity_y] 
	"""
	if len(path) < 2:
		raise ValueError("Path must have at least two points")

	# Segment vectors and directions
	segs = path[1:] - path[:-1]
	seg_lens = np.linalg.norm(segs, axis=1)
	seg_dirs = (segs.T / seg_lens).T

	# Time at each point 
	time_s = np.zeros(len(path))
	time_s[1:] = np.cumsum(seg_lens)
	total_time = time_s[-1]
	
	#trapesoid speed profile
	speeds = np.ones(len(path)) * target_velocity
	if ramp_up_time > 0:
		mask_up = time_s < ramp_up_time
		speeds[mask_up] = target_velocity * (time_s[mask_up] / ramp_up_time)
	if ramp_down_time > 0:
		mask_down = time_s > (total_time - ramp_down_time)
		speeds[mask_down] = target_velocity * ((total_time - time_s[mask_down]) / ramp_down_time)

	# Interpolated directions: avg of incoming/outgoing segments
	dirs = np.zeros_like(path)
	dirs[1:-1] = (seg_dirs[:-1] + seg_dirs[1:]) / 2
	dirs[1:-1] /= np.linalg.norm(dirs[1:-1], axis=1, keepdims=True)  # re-normalize
	dirs[0] = np.zeros(2)  # start at rest
	dirs[-1] = np.zeros(2)  # end at rest

	# Apply speed
	vels = dirs * speeds[:, None]
	return vels

		

	

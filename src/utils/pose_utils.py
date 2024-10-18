import quaternion as q
import numpy as np

def get_sim_location(position,rotation):
    x = -position[2]
    y = -position[0]
    axis = q.as_euler_angles(rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = q.as_euler_angles(rotation)[1]
    else:
        o = 2*np.pi - q.as_euler_angles(rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(o) + dy * np.cos(o)
    global_dy = dx * np.cos(o) - dy * np.sin(o)
    x += global_dy
    y += global_dx
    o += do
    if o > np.pi:
        o -= 2*np.pi
    return x, y, o

def pose_to_Rt(position, rotation):
    if isinstance(rotation, np.ndarray):
        rotation = q.from_float_array(rotation)
    rotation = q.as_rotation_matrix(rotation)
    Rt = np.eye(4)
    Rt[0:3,0:3] = rotation
    Rt[0:3,3] = position
    return np.linalg.inv(Rt)

def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

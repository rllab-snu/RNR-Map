import numpy as np
import cv2
import quaternion as q

def add_title(image, title, bg_color=(255,255,255)):
    # assume the image has RGB color (3 channels)
    title_im = np.full((30, image.shape[1], 3), bg_color, dtype=np.uint8)
    title_im = cv2.putText(title_im, title, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
    image = np.concatenate((title_im, image), axis=0)
    return image

def get_angle(rotation):
    euler = q.as_euler_angles(rotation)
    if (euler[0]%(2*np.pi)) < 0.1 or (euler[0]%(2*np.pi)) > 2*np.pi - 0.1:
        o = euler[1]
    else:
        o = 2*np.pi - euler[1]
    if o > np.pi:
        o -= 2 * np.pi
    return o

import torch
def add_agent_view_on_w(w_im, Rt, coordinate_scale, map_size, agent_color=(255,0,0),
                        agent_size=4, transparency=0.7, view_color=(255,255,255), view_size=20):
    if isinstance(Rt,torch.Tensor):
        Rt = Rt.squeeze().detach().cpu().numpy()
    x, _, y = np.linalg.inv(Rt)[:3, 3]
    agent_h = int(x / (coordinate_scale / 2.) * map_size / 2 + map_size / 2)
    agent_w = int(y / (coordinate_scale / 2.) * map_size / 2 + map_size/ 2)
    agent_o = get_angle(q.from_rotation_matrix(Rt[:3, :3]))
    a = - agent_o + np.pi
    ellipse_im = np.zeros_like(w_im)
    ellipse_im = cv2.ellipse(ellipse_im, (agent_w, agent_h), (view_size, view_size), 0, (a / np.pi * 180 - 45),
                             (a / np.pi * 180 + 45), view_color, -1)
    ellipse_mask = np.expand_dims((ellipse_im.sum(2) != 0) * transparency, 2)
    w_im = ((1 - ellipse_mask) * w_im + ellipse_mask * ellipse_im).astype(np.uint8)
    padd = max(agent_size//2,1)
    w_im[agent_h - padd:agent_h + padd, agent_w - padd:agent_w + padd] = agent_color
    return w_im

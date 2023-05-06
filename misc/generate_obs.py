import numpy as np 
import pygame 
from pygame import gfxdraw


def get_image_data(state, screen_dim=42):
    screen = pygame.Surface((screen_dim, screen_dim))
    
    surf = pygame.Surface((screen_dim, screen_dim))
    surf.fill((255, 255, 255))
    
    bound = 2.2
    scale = screen_dim / (bound * 2)
    offset = screen_dim // 2

    rod_length = 1 * scale
    rod_width = 0.2 * scale
    l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
    coords = [(l, b), (l, t), (r, t), (r, b)]
    transformed_coords = []
    for c in coords:
        c = pygame.math.Vector2(c).rotate_rad(state[0] + np.pi / 2)
        c = (c[0] + offset, c[1] + offset)
        transformed_coords.append(c)
    gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
    gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

    gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
    gfxdraw.filled_circle(
        surf, offset, offset, int(rod_width / 2), (204, 77, 77)
    )

    rod_end = (rod_length, 0)
    rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] + np.pi / 2)
    rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    gfxdraw.aacircle(
        surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )
    gfxdraw.filled_circle(
        surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )

    # drawing axle
    gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))
    
    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )
 
def process(ob, screen_dim=42):
    # if frame.size == 84 * 160 * 3:
    #     img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    # elif frame.size == 250 * 160 * 3:
    #     img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    # else:
    #     assert False, "Unknown resolution."
    img = ob.astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = np.reshape(img, [screen_dim, screen_dim, 1])
    ob = img.astype(np.uint8)
    ob = np.swapaxes(ob, 2, 0)
    return ob

def get_next_state(state, u, max_speed=8., m=1.0, l=1.0, g=9.81, dt=0.5, max_torque=2.0):
    th, thdot = state.squeeze()
    u = np.clip(u, -max_torque, max_torque)
    
    newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
    newthdot = np.clip(newthdot, max_speed, max_speed)
    newth = th + newthdot * dt

    new_state = np.array([newth, newthdot])
    return new_state

def get_next_obs(u, info):
    next_obs = []
    for idx in range(u.shape[0]):
        state = info[idx]
        action = u[idx]
        
        next_state = get_next_state(state, action)
        next_image = get_image_data(next_state)
        next_image = process(next_image)
        next_velocity = [next_state[1]]
        
        next_ob = [next_image, next_velocity]
        next_obs.append(next_ob)
    
    return next_obs

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def get_reward(state, u):
    th = state.squeeze()[:, 0].reshape(-1, 1)
    thdot = state.squeeze()[:, 1].reshape(-1, 1)
    costs = angle_normalize(th)**2 + 0.1 * thdot**2 + 0.001 * (u**2)
    return -costs
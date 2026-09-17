import numpy as np
from scipy.spatial.transform import Rotation as R

def propagate_ins_state(state: np.ndarray, u: np.ndarray, dt: float, gravity: float = 9.81) -> np.ndarray:
    p, v, euler = state[0:3], state[3:6], state[6:9]
    b_a, b_g = state[9:12], state[12:15]
    
    a_b = u[0:3] - b_a
    g_b = u[3:6] - b_g
    
    rot = R.from_euler('xyz', euler)
    delta_rot = R.from_rotvec(g_b * dt)
    rot_next = rot * delta_rot
    euler_next = rot_next.as_euler('xyz')
    
    a_n = rot.apply(a_b)
    a_n[2] -= gravity
    
    v_next = v + a_n * dt
    p_next = p + v * dt + 0.5 * a_n * (dt**2)
    
    return np.concatenate([p_next, v_next, euler_next, b_a, b_g])
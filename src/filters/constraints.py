import numpy as np
import scipy.special as sp
from scipy.spatial.transform import Rotation as R

def is_stationary(accel_window: np.ndarray, gyro_window: np.ndarray, 
                  base_accel_var: float, base_gyro_mag: float,
                  accel_mult: float = 1.5, gyro_mult: float = 1.5) -> bool:
    accel_var = np.var(accel_window, axis=0).sum()
    gyro_mag = np.linalg.norm(np.mean(gyro_window, axis=0))
    return bool(accel_var <= base_accel_var * accel_mult and gyro_mag <= base_gyro_mag * gyro_mult)

def hx_gnss(state: np.ndarray) -> np.ndarray:
    return np.array([state[0], state[1]])

def hx_nhc(state: np.ndarray) -> np.ndarray:
    v_nav = state[3:6]
    v_body = R.from_euler('xyz', state[6:9]).inv().apply(v_nav)
    return np.array([v_body[1], v_body[2]])

def hx_speed(state: np.ndarray) -> np.ndarray:
    v_nav = state[3:6]
    v_body = R.from_euler('xyz', state[6:9]).inv().apply(v_nav)
    return np.array([v_body[0]])

def chi_square_gate(nis: float, dof: int, confidence: float = 0.95) -> bool:
    p_value = sp.gammainc(dof / 2.0, nis / 2.0)
    return p_value <= confidence

def match_cnn_speed(t_curr: float, cnn_t: np.ndarray, cnn_v: np.ndarray, tol: float):
    """Find CNN speed if timestamp is within tolerance, no interpolation."""
    if len(cnn_t) == 0:
        return np.nan
    idx = np.searchsorted(cnn_t, t_curr)
    
    # Check bounds and nearest
    best_err = np.inf
    best_v = np.nan
    
    if idx < len(cnn_t):
        err = abs(cnn_t[idx] - t_curr)
        if err <= tol and err < best_err:
            best_err = err
            best_v = cnn_v[idx]
            
    if idx > 0:
        err = abs(cnn_t[idx-1] - t_curr)
        if err <= tol and err < best_err:
            best_err = err
            best_v = cnn_v[idx-1]
            
    return best_v
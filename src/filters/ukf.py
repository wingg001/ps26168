import numpy as np
from scipy.linalg import cholesky

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def state_diff(a, b):
    diff = a - b
    diff[6:9] = normalize_angle(diff[6:9])
    return diff

def state_mean(sigmas, Wm):
    x = np.sum(Wm[:, None] * sigmas, axis=0)
    for j in range(6, 9):
        s_sum = np.sum(Wm * np.sin(sigmas[:, j]))
        c_sum = np.sum(Wm * np.cos(sigmas[:, j]))
        x[j] = np.arctan2(s_sum, c_sum)
    return x

class ScaledUKF:
    def __init__(self, dim_x: int, alpha: float, beta: float, kappa: float):
        self.dim_x = dim_x
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.regularization_count = 0
        
        self.lam = alpha**2 * (dim_x + kappa) - dim_x
        self.c = dim_x + self.lam
        
        self.Wm = np.full(2 * dim_x + 1, 1.0 / (2 * self.c))
        self.Wc = np.full(2 * dim_x + 1, 1.0 / (2 * self.c))
        
        self.Wm[0] = self.lam / self.c
        self.Wc[0] = self.lam / self.c + (1 - alpha**2 + beta)

    def predict(self, fx, dt, u):
        sigmas = self._generate_sigmas()
        sigmas_f = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            sigmas_f[i] = fx(sigmas[i], u, dt)
            
        self.x = state_mean(sigmas_f, self.Wm)
        
        y = np.zeros_like(sigmas_f)
        for i in range(len(sigmas_f)):
            y[i] = state_diff(sigmas_f[i], self.x)
            
        self.P = y.T @ np.diag(self.Wc) @ y + self.Q
        self._symmetrize_and_ensure_psd()

    def update(self, z, hx, R_meas, gating_func=None):
        sigmas = self._generate_sigmas()
        dim_z = len(z)
        sigmas_h = np.zeros((len(sigmas), dim_z))
        for i in range(len(sigmas)):
            sigmas_h[i] = hx(sigmas[i])
            
        zp = np.sum(self.Wm[:, None] * sigmas_h, axis=0)
        
        y_h = sigmas_h - zp
        x_diff = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            x_diff[i] = state_diff(sigmas[i], self.x)
            
        Pzz = y_h.T @ np.diag(self.Wc) @ y_h + R_meas
        Pxz = x_diff.T @ np.diag(self.Wc) @ y_h
        
        try:
            inv_Pzz = np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            # Numerical failure is represented by NaN NIS so the caller can
            # distinguish it from a deliberate statistical rejection.
            return False, float("nan"), False
            
        y_res = z - zp
        nis = float(y_res.T @ inv_Pzz @ y_res)
        
        accepted = True
        if gating_func is not None:
            if not gating_func(nis, dim_z):
                accepted = False
                return False, nis, accepted
                
        K = Pxz @ inv_Pzz
        self.x = self.x + K @ y_res
        self.x[6:9] = normalize_angle(self.x[6:9])
        
        self.P = self.P - K @ Pzz @ K.T
        self._symmetrize_and_ensure_psd()
        return True, nis, accepted

    def _symmetrize_and_ensure_psd(self):
        # 1. Symmetrize
        self.P = 0.5 * (self.P + self.P.T)
        
        # 2. Check Eigenvalues for Positive Semi-Definiteness (PSD)
        vals, vecs = np.linalg.eigh(self.P)
        if np.any(vals < 1e-8):
            self.regularization_count += 1
            vals = np.maximum(vals, 1e-8)
            self.P = vecs @ np.diag(vals) @ vecs.T

    def _generate_sigmas(self):
        try:
            U = cholesky(self.c * self.P, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if strict Cholesky fails despite PSD check
            vals, vecs = np.linalg.eigh(self.c * self.P)
            vals = np.maximum(vals, 1e-8)
            U = vecs @ np.diag(np.sqrt(vals))
            
        sigmas = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigmas[0] = self.x
        for k in range(self.dim_x):
            sigmas[k + 1] = self.x + U[:, k]
            sigmas[self.dim_x + k + 1] = self.x - U[:, k]
        return sigmas
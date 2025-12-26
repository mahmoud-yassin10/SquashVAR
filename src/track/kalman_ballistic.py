import numpy as np
from filterpy.kalman import KalmanFilter

class BallisticKF:
    """
    2D image-space state: [x, y, vx, vy].  dt in frames; add gravity on vy if wanted.
    """
    def __init__(self, dt=1.0, process_std=5.0, meas_std=3.0, gravity_px=0.0):
        self.dt = dt
        self.g  = gravity_px
        self.f  = KalmanFilter(dim_x=4, dim_z=2)
        self.f.F = np.array([[1,0,dt,0],
                             [0,1,0,dt],
                             [0,0,1, 0],
                             [0,0,0, 1]], dtype=float)
        self.f.H = np.array([[1,0,0,0],
                             [0,1,0,0]], dtype=float)
        q = process_std**2
        r = meas_std**2
        self.f.Q = np.eye(4) * q
        self.f.R = np.eye(2) * r
        self.f.P = np.eye(4) * 100.0  # large init uncertainty

    def init_state(self, x, y, vx=0.0, vy=0.0):
        self.f.x = np.array([x, y, vx, vy], dtype=float)

    def predict(self):
        # add ballistic gravity on vy
        self.f.F[1,3] = self.dt
        self.f.F[0,2] = self.dt
        self.f.predict()
        if self.g != 0.0:
            self.f.x[3] += self.g * self.dt  # vy += g*dt
        return self.f.x.copy()

    def update(self, zxy):
        self.f.update(np.array(zxy, dtype=float))
        return self.f.x.copy()

def reflect_velocity_on_plane(state, normal, restitution=0.85):
    """
    state: [x,y,vx,vy]; reflect velocity vector across plane normal.
    """
    v = np.array([state[2], state[3]], dtype=float)
    n = np.array(normal, dtype=float); n = n/np.linalg.norm(n)
    v_ref = v - 2*np.dot(v, n)*n
    v_ref *= restitution
    return np.array([state[0], state[1], v_ref[0], v_ref[1]])

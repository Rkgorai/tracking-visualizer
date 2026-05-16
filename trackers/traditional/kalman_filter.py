import numpy as np


class KalmanFilter:
    """Kalman Filter for 2D bounding box tracking."""

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        # State: [x, y, w, h, vx, vw, vy, vh]
        self.state_dim = 8
        self.measure_dim = 4

        # State transition matrix
        self.F = np.eye(self.state_dim)
        self.F[0, 4] = dt  # x += vx * dt
        self.F[1, 6] = dt  # y += vy * dt
        self.F[2, 5] = dt  # w += vw * dt
        self.F[3, 7] = dt  # h += vh * dt

        # Observation matrix
        self.H = np.zeros((self.measure_dim, self.state_dim))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # w
        self.H[3, 3] = 1  # h

        # Process and measurement noise
        self.Q = np.eye(self.state_dim) * 1e-4
        self.R = np.eye(self.measure_dim) * 0.1

        self.x = np.zeros(self.state_dim)  # State
        self.P = np.eye(self.state_dim)     # Covariance

    def initiate(self, measurement: np.ndarray) -> None:
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = measurement
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        self.P = np.eye(self.state_dim)

    def predict(self) -> np.ndarray:
        """Predict next state and return bounding box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._get_bbox()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state with new measurement."""
        z = measurement  # [x1, y1, x2, y2] -> convert to center
        z_h = self.H @ self.x
        z = np.array([(z[0] + z[2]) / 2, (z[1] + z[3]) / 2, z[2] - z[0], z[3] - z[1]])

        y = z - z_h  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        return self._get_bbox()

    def _get_bbox(self) -> np.ndarray:
        """Convert state to [x1, y1, x2, y2] format."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanFilter3D(KalmanFilter):
    """Extended Kalman Filter for 3D tracking with depth."""

    def __init__(self, dt: float = 1.0):
        # State: [x, y, z, w, h, d, vx, vy, vz, vw, vh, vd]
        self.state_dim = 12
        self.measure_dim = 4

        self.dt = dt
        self.F = np.eye(self.state_dim)
        for i in range(6):
            self.F[i, 6 + i] = dt

        self.H = np.zeros((self.measure_dim, self.state_dim))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(self.state_dim) * 1e-4
        self.R = np.eye(self.measure_dim) * 0.1

        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)
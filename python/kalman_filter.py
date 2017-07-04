from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import math
import numpy as np

class KalmanFilter:
    def __init__(self):
        self.filter = create_filter()

    def predict(self, dt):
        self.filter.predict(dt)

        self.x = self.filter.x

    def update(self, z, dt):
        self.filter.predict(dt)
        self.filter.update(z)

        self.x = self.filter.x

def normalize_angle(angle):
    while angle > math.pi:
        angle -= math.pi
    while angle < -math.pi:
        angle += math.pi

    return angle

def fx(x, dt):
    p_x = x[0]
    p_y = x[1]
    v   = x[2]
    yaw = x[3]
    yawd = x[4]

    nu_a = 2
    nu_yawdd = 0.2

    # Avoid division by zero.
    if math.fabs(yawd) > 0.001:
        px_p = p_x + v / yawd * (math.sin(yaw + yawd * dt) - math.sin(yaw))
        py_p = p_y + v / yawd * (math.cos(yaw) - math.cos(yaw + yawd * dt))
    else:
        px_p = p_x + v * dt * math.cos(yaw)
        py_p = p_y + v * dt * math.sin(yaw)

    v_p = v
    yaw_p = yaw + yawd * dt
    yawd_p = yawd

    # Add noise.
    px_p = px_p + 0.5 * nu_a * dt * dt * math.cos(yaw)
    py_p = py_p + 0.5 * nu_a * dt * dt * math.sin(yaw)
    v_p = v_p + nu_a * dt

    yaw_p = yaw_p + 0.5 * nu_yawdd * dt * dt
    yawd_p = yawd_p + nu_yawdd * dt

    yaw_p = normalize_angle(yaw_p)

    return np.array([px_p,
                     py_p,
                     v_p,
                     yaw_p,
                     yawd_p])

def hx(x):
    return x[0:2]

def create_filter():
    dim_x = 5
    points = MerweScaledSigmaPoints(dim_x, alpha = 1e-3, beta = 2, kappa = 0.0)

    ukf = UKF(dim_x = dim_x, dim_z = 2, dt = 0.1, hx = hx, fx = fx, points = points)
    ukf.Q *= np.identity(5) * 0.2
    std_las = 0.15
    ukf.R *= std_las * std_las
    ukf.x = np.zeros(5)
    ukf.P *= 0.2

    return ukf


if __name__ == "__main__":
    filter = KalmanFilter()
    for i in range(100):
        z = -np.ones(2) * i * 0.1
        filter.update(z, dt = 0.1)
        print(filter.x)
    print('Start predicting...')
    for i in range(100):
        filter.predict(dt = 0.1)
        print(filter.x)

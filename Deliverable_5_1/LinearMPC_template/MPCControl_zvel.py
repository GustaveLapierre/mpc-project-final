import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import place_poles

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    def _setup_controller(self) -> None:
        self.U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([80.0 - self.us[0], -(40.0 - self.us[0])], dtype=float),
        )

        self.X = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([20.0, 20.0]),
        )

        self.Q = np.diag([5.0])
        self.R = np.diag([1.0])
        self.S = np.diag([1000.0] * self.X.A.shape[0])  # Slack variable penalty
        self.s = 2000.0  # Slack variable weight

        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K

        self.x_var = cp.Variable((self.nx, self.N + 1))
        self.u_var = cp.Variable((self.nu, self.N))
        self.x0_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)
        # Slack variables
        self.s_var = cp.Variable((self.X.A.shape[0], self.N + 1), nonneg=True)

        cost = 0
        constraints = [self.x_var[:, 0] == self.x0_param]

        for k in range(self.N):
            constraints += [
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            ]
            constraints += [
                self.X.A @ (self.x_var[:, k] + self.x_ref_param) <= self.X.b
            ]
            constraints += [
                self.U.A @ (self.u_var[:, k] + self.u_ref_param) <= self.U.b
            ]
            cost += cp.quad_form(self.x_var[:, k], self.Q)
            cost += cp.quad_form(self.u_var[:, k], self.R)
            # Cost for the slack variables
            cost += cp.quad_form(self.s_var[:, k], self.S) + self.s * cp.norm1(self.s_var[:, k])

        cost += cp.quad_form(self.x_var[:, -1], self.Qf)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

        self.setup_estimator()

    def get_u(self, x0, x_target=None, u_target=None):
        if x_target is None:
            x_target = np.zeros(self.nx)
        if u_target is None:
            u_target = np.zeros(self.nu)

        self.update_estimator(x0, self.u_prev)

        if np.abs(x_target[0]) > 1e-9:
            A = self.A[0, 0]
            B = self.B[0, 0]
            u_abs = self.us[0] + (1.0 - A) / B * x_target[0]
            u_ref = u_abs - self.us[0]
        else:
            u_ref = 0.0

        u_ref_with_d = u_ref - self.d_estimate

        x0_est = np.array([self.x_hat[0]])

        self.x0_param.value = x0_est - x_target
        self.x_ref_param.value = x_target
        self.u_ref_param.value = np.array([u_ref_with_d])

        self.ocp.solve(solver=cp.CLARABEL, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u_out = np.array([self.us[0] + u_ref_with_d])
            self.u_prev = u_ref_with_d
            return (
                u_out,
                np.tile(x_target[:, None], (1, self.N + 1)),
                np.tile(np.array([[self.us[0] + u_ref_with_d]]), (1, self.N)),
            )

        u_mpc = self.u_var.value[0, 0]
        self.u_prev = u_mpc + u_ref_with_d

        return (
            self.u_var.value[:, 0] + self.us[0] + u_ref_with_d,
            self.x_var.value + x_target[:, None],
            self.u_var.value + self.us[0] + u_ref_with_d,
        )

    def setup_estimator(self): # Luenberger observer
        A_scalar = self.A[0, 0]
        B_scalar = self.B[0, 0]
        Bd_scalar = B_scalar

        self.A_aug = np.array([[A_scalar, Bd_scalar], [0.0, 1.0]])
        self.B_aug = np.array([[B_scalar], [0.0]])
        self.C_aug = np.array([[1.0, 0.0]])

        desired_poles = np.array([0.6, 0.7])
        result = place_poles(self.A_aug.T, self.C_aug.T, desired_poles)
        self.L = result.gain_matrix.T

        self.x_hat = np.array([0.0, 0.0])
        self.d_estimate = 0.0
        self.u_prev = 0.0
        self.d_history = []
        self.first_call = True

    def update_estimator(self, x_meas: np.ndarray, u_prev: float) -> None:
        y_meas = x_meas[0] if isinstance(x_meas, np.ndarray) else x_meas

        if self.first_call:
            self.x_hat[0] = y_meas
            self.first_call = False
            return

        x_pred = self.A_aug @ self.x_hat + self.B_aug.flatten() * u_prev

        y_pred = self.C_aug @ x_pred
        innovation = y_meas - y_pred[0]

        self.x_hat = x_pred + self.L.flatten() * innovation
        self.d_estimate = self.x_hat[1]
        self.d_history.append(self.d_estimate)

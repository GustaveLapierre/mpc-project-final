import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

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

        self.Q = np.diag([15.0])
        self.R = np.diag([0.5])
        self.S = np.diag([1000.0] * self.X.A.shape[0])  # Slack variable penalty
        self.s = 2000.0  # Slack variable weight

        K, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K

        A_cl = self.A + self.B @ self.K
        KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b)
        self.O_inf = self.max_invariant_set(A_cl, self.X.intersect(KU))

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

        constraints += [
            self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b
        ]
        cost += cp.quad_form(self.x_var[:, -1], self.Qf)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(self, x0, x_target=None, u_target=None):
        if x_target is None:
            x_target = np.zeros(self.nx)
        if u_target is None:
            u_target = np.zeros(self.nu)

        if np.abs(x_target[0]) > 1e-9:
            A = self.A[0, 0]
            B = self.B[0, 0]
            u_abs = self.us[0] + (1.0 - A) / B * x_target[0]
            u_ref = u_abs - self.us[0]
        else:
            u_ref = 0.0

        self.x0_param.value = x0 - x_target
        self.x_ref_param.value = x_target
        self.u_ref_param.value = np.array([u_ref])

        self.ocp.solve(solver=cp.CLARABEL, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            return (
                np.array([self.us[0] + u_ref]),
                np.tile(x_target[:, None], (1, self.N + 1)),
                np.tile(np.array([[self.us[0] + u_ref]]), (1, self.N)),
            )

        return (
            self.u_var.value[:, 0] + self.us[0] + u_ref,
            self.x_var.value + x_target[:, None],
            self.u_var.value + self.us[0] + u_ref,
        )

    def setup_estimator(self):
        self.d_estimate = 0.0
        self.d_gain = 0.0

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        return

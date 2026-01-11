import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        self.U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([0.26, 0.26]),
        )

        F = np.eye(3)
        F = np.vstack((F, -F))
        f = np.array([10.0, 0.1745, 20.0, 10.0, 0.1745, 20.0])
        self.X = Polyhedron.from_Hrep(F, f)

        self.Q = np.diag([10.0, 150.0, 2.0])
        self.R = np.diag([20.0])

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

        self.x0_param.value = x0 - x_target
        self.x_ref_param.value = x_target
        self.u_ref_param.value = u_target

        self.ocp.solve(solver=cp.CLARABEL, verbose=False)

        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            return (
                u_target,
                np.tile(x_target[:, None], (1, self.N + 1)),
                np.tile(u_target[:, None], (1, self.N)),
            )

        return (
            self.u_var.value[:, 0] + u_target,
            self.x_var.value + x_target[:, None],
            self.u_var.value + u_target[:, None],
        )

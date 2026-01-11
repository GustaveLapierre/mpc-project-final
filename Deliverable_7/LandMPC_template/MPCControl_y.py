import numpy as np
import cvxpy as cp
from control import dlqr

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    """Nominal MPC for y-position control.
    State: [wx, alpha, vy, y] - angular velocity x, roll angle, velocity y, position y
    Input: [d1] - roll gimbal angle
    """
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        u_max = np.deg2rad(15) - self.us[0]
        u_min = -np.deg2rad(15) - self.us[0]
        
        self.Q = np.diag([1.0, 1.0, 1.0, 10.0])  # Penalize position more
        self.R = np.diag([0.1])
        
        _, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        
        self.s_penalty = 1000.0
        
        self.x_var = cp.Variable((self.N + 1, self.nx))
        self.u_var = cp.Variable((self.N, self.nu))
        self.x0_param = cp.Parameter(self.nx)
        self.slack = cp.Variable((self.N, 2), nonneg=True)
        
        cost = 0
        constraints = [self.x_var[0] == self.x0_param]
        
        for k in range(self.N):
            constraints.append(self.x_var[k+1] == self.A @ self.x_var[k] + self.B @ self.u_var[k])
            constraints.append(self.u_var[k] <= u_max + self.slack[k, 0])
            constraints.append(self.u_var[k] >= u_min - self.slack[k, 1])
            cost += cp.quad_form(self.x_var[k], self.Q)
            cost += cp.quad_form(self.u_var[k], self.R)
            cost += self.s_penalty * cp.sum(self.slack[k])
        
        cost += cp.quad_form(self.x_var[-1], self.Qf)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0_delta = x0 - self.xs
        self.x0_param.value = x0_delta
        
        self.ocp.solve(solver=cp.CLARABEL, verbose=False)
        
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u0 = self.us.copy()
            x_traj = np.tile(self.xs[:, None], (1, self.N + 1))
            u_traj = np.tile(self.us[:, None], (1, self.N))
            return u0, x_traj, u_traj
        
        u0 = self.u_var[0].value + self.us
        x_traj = self.x_var.value.T + self.xs[:, None]
        u_traj = self.u_var.value.T + self.us[:, None]
        
        return u0, x_traj, u_traj

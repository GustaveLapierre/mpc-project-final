import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        u_min = 40.0 - self.us[0]
        u_max = 80.0 - self.us[0]
        self.U = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([u_max, -u_min])
        )
        # TODO params to tune
        vz_max = 15.0
        z_margin = 3.0  
        z_min_delta = -(self.xs[1] - z_margin)
        z_max_delta = 15.0
        
        self.X = Polyhedron.from_Hrep(
            np.array([
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]),
            np.array([vz_max, vz_max, -z_min_delta, z_max_delta])
        )
        
        w_min, w_max = -15.0, 5.0
        W_input = Polyhedron.from_Hrep(
            np.array([[1.0], [-1.0]]),
            np.array([w_max, -w_min])
        )
        self.W = W_input.affine_map(self.B)
        
        # T ODO params to tune
        self.Q = np.diag([10.0, 100.0])
        self.R = np.diag([0.1])
        K_lqr, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K_lqr
        self.A_cl = self.A + self.B @ self.K
        
        self.E = self._min_robust_invariant_set(self.A_cl, self.W)
        
        self.X_tilde = self.X - self.E
        self.KE = self.E.affine_map(self.K)
        self.U_tilde = self.U - self.KE
        
        X_tilde_and_KU_tilde = self.X_tilde.intersect(
            Polyhedron.from_Hrep(self.U_tilde.A @ self.K, self.U_tilde.b)
        )
        self.Xf_tilde = self.max_invariant_set(self.A_cl, X_tilde_and_KU_tilde)
        
        self.z_var = cp.Variable((self.N + 1, self.nx))
        self.v_var = cp.Variable((self.N, self.nu))
        self.x0_param = cp.Parameter(self.nx)
        
        cost = 0
        for k in range(self.N):
            cost += cp.quad_form(self.z_var[k], self.Q)
            cost += cp.quad_form(self.v_var[k], self.R)
        cost += cp.quad_form(self.z_var[-1], self.Qf)
        
        constraints = []
        constraints.append(self.E.A @ (self.x0_param - self.z_var[0]) <= self.E.b)
        constraints.append(self.z_var[1:].T == self.A @ self.z_var[:-1].T + self.B @ self.v_var.T)
        constraints.append(self.X_tilde.A @ self.z_var[:-1].T <= self.X_tilde.b.reshape(-1, 1))
        constraints.append(self.U_tilde.A @ self.v_var.T <= self.U_tilde.b.reshape(-1, 1))
        constraints.append(self.Xf_tilde.A @ self.z_var[-1] <= self.Xf_tilde.b)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _min_robust_invariant_set(A_cl: np.ndarray, W: Polyhedron, max_iter: int = 100) -> Polyhedron:
        """Compute minimal robust positively invariant (mRPI) set."""
        nx = A_cl.shape[0]
        Omega = W
        for itr in range(1, max_iter + 1):
            A_cl_power = np.linalg.matrix_power(A_cl, itr)
            if np.linalg.norm(A_cl_power, ord=2) < 1e-2:
                print(f'mRPI set converged after {itr} iterations.')
                return Omega
            Omega_next = Omega + A_cl_power @ W
            Omega_next.minHrep()
            Omega = Omega_next
        print(f'mRPI set did NOT converge after {max_iter} iterations.')
        return Omega

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0_delta = x0 - self.xs
        self.x0_param.value = x0_delta
        
        self.ocp.solve(solver=cp.CLARABEL, verbose=False)
        
        if self.ocp.status not in ("optimal", "optimal_inaccurate"):
            u_delta = self.K @ x0_delta
            u0 = np.clip(u_delta + self.us, 40.0, 80.0)
            x_traj = np.tile(self.xs[:, None], (1, self.N + 1))
            u_traj = np.tile(self.us[:, None], (1, self.N))
            return u0, x_traj, u_traj
        
        z0 = self.z_var[0].value
        v0 = self.v_var[0].value
        u_delta = v0 + self.K @ (x0_delta - z0)
        u0 = np.clip(u_delta + self.us, 40.0, 80.0)
        
        x_traj = self.z_var.value.T + self.xs[:, None]
        u_traj = self.v_var.value.T + self.us[:, None]
        
        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        self.d_estimate = ...
        self.d_gain = ...

        pass

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        pass

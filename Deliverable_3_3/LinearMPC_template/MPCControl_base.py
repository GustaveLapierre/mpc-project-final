import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    x_ids: np.ndarray
    u_ids: np.ndarray

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = float(Ts)
        self.H = float(H)
        self.N = int(np.round(H / Ts))

        self.nx = int(self.x_ids.shape[0])
        self.nu = int(self.u_ids.shape[0])

        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T

        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, self.Ts)

        self.xs = xs[self.x_ids].copy()
        self.us = us[self.u_ids].copy()

        self._setup_controller()

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_d, B_d, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_d, B_d

    @staticmethod
    def max_invariant_set(A_cl: np.ndarray, X: Polyhedron, max_iter: int = 50) -> Polyhedron:
        O = X
        for itr in range(1, max_iter + 1):
            Oprev = O
            F, f = O.A, O.b
            O = Polyhedron.from_Hrep(
                np.vstack((F, F @ A_cl)),
                np.hstack((f, f)).reshape((-1,))
            )
            O.minHrep(True)
            _ = O.Vrep
            if O == Oprev:
                return O
        return O

    def _setup_controller(self) -> None:
        raise NotImplementedError

    def get_u(
        self,
        x0: np.ndarray,
        x_target: np.ndarray = None,
        u_target: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

import numpy as np
import casadi as ca
from typing import Tuple
from control import dlqr


class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    def __init__(self, rocket, H, xs, us):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """        
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]

        self.rocket = rocket
        self.H = H
        self.xs = xs
        self.us = us

        self.nx = rocket.nx
        self.nu = rocket.nu
        self.dt = rocket.Ts
        self.N = int(H / rocket.Ts)

        self._setup_controller()

    # Runge-Kutta 4 integrator
    def rk4(self, f, x, u, h):
        k1 = h * f(x, u)
        k2 = h * f(x + k1 / 2, u)
        k3 = h * f(x + k2 / 2, u)
        k4 = h * f(x + k3, u)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def f_d_casadi(self, x, u):
        """Discrete dynamics using RK4 integration for CASADI variables."""
        return self.rk4(self.f, x, u, self.dt)

    def _setup_controller(self) -> None:
        
        # Tunable weights:
        self.Q = np.diag([10.0, 10.0, 20.0,        # w_x,  w_y,  w_z
                          1.0, 1.0, 50.0,           # alpha, beta, gamma
                          10.0, 10.0, 10.0,        # v_x, v_y, v_z
                          100.0, 100.0, 100.0])    # p_x, p_y, p_z
        self.R = np.diag([100.0, 100.0, 0.1, 0.1]) # delta1, delta2, P_avg, P_diff

        # Linearize dynamics at the steady state to calculate the terminal cost matrix Qf
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        jac_x = ca.Function('jac_x', [x, u], [ca.jacobian(self.f(x, u), x)])
        jac_u = ca.Function('jac_u', [x, u], [ca.jacobian(self.f(x, u), u)])
        self.A = jac_x(self.xs, self.us).toarray()
        self.B = jac_u(self.xs, self.us).toarray()
        _, self.Qf, _ = dlqr(self.A, self.B, self.Q, self.R)

        
        self.ocp = ca.Opti()

        # Decision variables
        self.X = self.ocp.variable(self.nx, self.N+1)  
        self.U = self.ocp.variable(self.nu, self.N)  

        self.X0 = self.ocp.parameter(self.nx, 1)  # Initial state parameter       

        # Angular velocities about the body axes
        w_x = self.X[0, :] 
        w_y = self.X[1, :] 
        w_z = self.X[2, :]

        # Euler angles
        alpha = self.X[3, :]  
        beta = self.X[4, :]
        gamma = self.X[5, :]

        # Velocities in the world frame
        v_x = self.X[6, :]  
        v_y = self.X[7, :]
        v_z = self.X[8, :]

        # Positions in the world frame
        p_x = self.X[9, :]
        p_y = self.X[10, :]
        p_z = self.X[11, :]
        
        # Objective function
        self.ocp.minimize(
            sum((self.X[:,k]-self.xs).T @ self.Q @ (self.X[:,k]-self.xs) + 
                 (self.U[:,k]-self.us).T @ self.R @ (self.U[:,k]-self.us) 
                 for k in range(self.N)) +
            (self.X[:,self.N]-self.xs).T @ self.Qf @ (self.X[:,self.N]-self.xs)
        )

        # Initial conditions       
        self.ocp.subject_to(self.X[:,0] == self.X0)
        # Dynamic constraints
        for k in range(self.N):
            self.ocp.subject_to(self.X[:, k+1] == self.f_d_casadi(self.X[:,k], self.U[:,k]))
        
        # Beta constraints
        beta_max = np.deg2rad(80) 
        beta_min = -np.deg2rad(80) 
        self.ocp.subject_to(self.ocp.bounded(beta_min, beta, beta_max))

        # Z position constraints
        z_min = 0.0 
        self.ocp.subject_to(p_z >= z_min)

        # Input constraints
        U_min = np.array([np.deg2rad(-15), np.deg2rad(-15), 10.0, -20.0]) 
        U_max = np.array([np.deg2rad(15), np.deg2rad(15), 90.0, 20.0]) 
        self.ocp.subject_to(self.ocp.bounded(U_min[:, None], self.U, U_max[:, None]))

        # Set solver options
        options = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                "tol": 1e-3,
            },
        }
        self.ocp.solver("ipopt", options)

        

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Set initial state parameter
        self.ocp.set_value(self.X0, x0)
        
        try:
            sol = self.ocp.solve()

            u0 = sol.value(self.U[:,0])
            x_ol = sol.value(self.X)
            u_ol = sol.value(self.U)
            t_ol = t0 + np.arange(self.N+1) * self.dt

        except Exception as e:
            print(f"WARNING: NMPC solver failed at time {t0:.2f}s with error: {e}")
            print("Debug - Last values before crash:")
            try:
                print("U_opt:", self.ocp.debug.value(self.U))
                print("X_opt:", self.ocp.debug.value(self.X))
            except:
                pass
            u0 = self.us
            x_ol = np.tile(self.xs[:, None], (1, self.N + 1))
            u_ol = np.tile(self.us[:, None], (1, self.N))
            t_ol = t0 + np.arange(self.N+1) * self.dt

        return u0, x_ol, u_ol, t_ol
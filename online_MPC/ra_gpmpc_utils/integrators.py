import torch
import gpytorch
import numpy as np
from zero_order_gpmpc.models import ResidualModel


# Time integration of a given ode_dynamics, using Runge Kutta of Euler for time step dt.
class TimeIntegrator(torch.nn.Module):
    def __init__(self, ode_dynamics, dt, method="RK4"):
        super().__init__()
        self.ode_dynamics = ode_dynamics
        self.dt = dt
        self.is_eliminate_GP = False    # set to true when using GP model elimination online
        self.w_bar_gp_idx_list = []     # store which GP model index is used to compute w_bar for all time integration steps  
        self.method = method.lower()    # integration method ("euler" or "rk4")
        
        # Check for valid method
        if self.method not in ["euler", "rk4"]:
            raise ValueError("Invalid method. Only 'Euler' and 'RK4' are supported.") 
    
    def runge_kutta(self, x, u):
        k1 = self.ode_dynamics(x, u)
        if self.is_eliminate_GP:
            self.w_bar_gp_idx_list.extend(self.ode_dynamics.estimate_model.last_w_bar_gp_idx)   

        k2 = self.ode_dynamics(x + 0.5 * self.dt * k1, u)
        if self.is_eliminate_GP:
            self.w_bar_gp_idx_list.extend(self.ode_dynamics.estimate_model.last_w_bar_gp_idx)

        k3 = self.ode_dynamics(x + 0.5 * self.dt * k2, u)
        if self.is_eliminate_GP:
            self.w_bar_gp_idx_list.extend(self.ode_dynamics.estimate_model.last_w_bar_gp_idx)

        k4 = self.ode_dynamics(x + self.dt * k3, u)
        if self.is_eliminate_GP:
            self.w_bar_gp_idx_list.extend(self.ode_dynamics.estimate_model.last_w_bar_gp_idx)

        x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next
    
    def explicit_euler(self, x, u):
        k1 = self.ode_dynamics(x, u)
        if self.is_eliminate_GP:
            self.w_bar_gp_idx_list.extend(self.ode_dynamics.estimate_model.last_w_bar_gp_idx)  

        x_next = x + self.dt * k1

        return x_next

    # forward method: tailored for SQP of zoroAcados. In particular:
    # - y has size n_shouting_nodes \times (nx + nu). 
    # - nx is the total number of states (system and tube), and nu is the total number of inputs (system and potential additional decision variables)
    # - the integrator computes the next step of each shouting nodes separately, and store the results in the matrix x_next_matrix
    def forward(self, y):                                                       
        x_next_matrix = torch.zeros(y.shape[0],self.ode_dynamics.nx)
        self.w_bar_gp_idx_list = []     # re-initialise the list
        n_shooting_nodes = y.shape[0]
        # reshape input to have n_shooting_nodes x nx and n_shooting_nodes x nu
        x_at_shooting_nodes = torch.reshape(y[:,0:self.ode_dynamics.nx],(n_shooting_nodes,self.ode_dynamics.nx))
        u_at_shooting_nodes = torch.reshape(y[:,self.ode_dynamics.nx:],(n_shooting_nodes,self.ode_dynamics.nu))

        if self.method == "euler":
            x_next_matrix = self.explicit_euler(x_at_shooting_nodes,u_at_shooting_nodes)
        else:   # default is RK4
            x_next_matrix = self.runge_kutta(x_at_shooting_nodes,u_at_shooting_nodes)

        return x_next_matrix
    
# Residual used by zero-order GP MPC framework. The residual is integrating the state of the systems using Runge Kutta or Euler, using the TimeIntegrator method
class ResidualTimeIntegration(ResidualModel):
    def __init__(self, system_time_integrator):
        self.system_time_integrator = system_time_integrator
        self.info = {}  # dictionary to store additional information if needed

        self.to_tensor = lambda X: torch.Tensor(X)
        self.to_numpy = lambda T: T.numpy()
        
        if torch.cuda.is_available():
            self.cuda_is_available = True
        else:
            self.cuda_is_available = False

        def mean_fun_sum(y):  
            with gpytorch.settings.fast_pred_var():
                return self.system_time_integrator(y).sum(dim=0)
        
        self._mean_fun_sum = mean_fun_sum

    def evaluate(self,y):
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=False)
            with torch.no_grad():
                self.predictions = self.system_time_integrator(y_tensor)
        return self.to_numpy(self.predictions)

    def jacobian(self,y):
        with gpytorch.settings.fast_pred_var():
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=True)
            mean_dy = torch.autograd.functional.jacobian(
                self._mean_fun_sum, 
                y_tensor
            )
        return self.to_numpy(mean_dy)

    def value_and_jacobian(self,y):
        with gpytorch.settings.fast_pred_var(): 
            y_tensor = torch.autograd.Variable(self.to_tensor(y), requires_grad=True)
            with torch.no_grad():
                predictions = self.system_time_integrator(y_tensor)
            with torch.no_grad():   # added this
                jacobian_matrix = torch.autograd.functional.jacobian(
                    self._mean_fun_sum, 
                    y_tensor
                )

        return predictions , jacobian_matrix

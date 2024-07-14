import torch
import gpytorch

## -------------------------------------------------------------------------------------------------------------------
##
## DYNAMICS WITHOUT GP ESTIMATES
##
## -------------------------------------------------------------------------------------------------------------------

# Dynamics of a linear system, without GP
# Note: the forward method describes the time derivative of the system, evaluated at all shooting nodes at once
class LinearSystemNoGP(torch.nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B
        self.nx_sys = A.shape[1]    # number of system states
        self.nu_sys = B.shape[1]    # number of system inputs

    # Note: x and u should be torch tensors of shape (n_shooting_nodes,nx) and (n_shooting_nodes,nu)
    def forward(self, x, u):
        n_shooting_nodes = x.shape[0]
        A_batch = self.A.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nx)
        B_batch = self.B.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nu)
        Ax = torch.bmm(A_batch, x.unsqueeze(2)).squeeze(2)  # brings x to shape (n_shooting_nodes,nx,1)
        Bu = torch.bmm(B_batch, u.unsqueeze(2)).squeeze(2)  # brings u to shape (n_shooting_nodes,nu,1)

        # shape of output is (n_shouting_nodes,nx)
        if Ax.shape != Bu.shape:
            raise Exception(f"Ax and Bu do not have the same shape.")
        return  Ax + Bu


# Dynamics of a nonlinear system defined by its function f and its input matrix B, without GP
# Note: the forward method describes the time derivative of the system, evaluated at all shooting nodes at once
class NonlinearSystemNoGP(torch.nn.Module):
    def __init__(self, B, f_fcn):
        super().__init__()
        self.B = B
        self.f_fcn = f_fcn
        self.nx_sys = B.shape[0]    # number of system states
        self.nu_sys = B.shape[1]    # number of system inputs

    # Note: x and u should be torch tensors of shape (n_shooting_nodes,nx) and (n_shooting_nodes,nu)
    def forward(self, x, u):
        n_shooting_nodes = x.shape[0]

        fx = self.f_fcn(x)
        B_batch = self.B.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nu)
        Bu = torch.bmm(B_batch, u.unsqueeze(2)).squeeze(2)  # brings u to shape (n_shooting_nodes,nu,1)

        # shape of output is (n_shouting_nodes,nx)
        if fx.shape != Bu.shape:
            raise Exception(f"f(x) and Bu do not have the same shape.")
        return  fx + Bu

# Dynamics of the tube, without GP 
# Note: the forward method describes the time derivative of the tube, evaluated at all shooting nodes at once
class TubeDynamicsNoGP(torch.nn.Module):
    def __init__(self,rho,max_norm_d_M):
        super().__init__()
        self.rho = torch.tensor([[-rho]])
        self.max_norm_d_M = torch.tensor([[max_norm_d_M]])
        self.ndelta = 1     # tube state is always 1-dimensional

    # Note: delta is a torch tensor of shape (n_shooting_nodes,1)
    def forward(self, delta): 
        n_shooting_nodes = delta.shape[0]
        rho_batch = self.rho.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,1,1)
        Adelta = torch.bmm(rho_batch, delta.unsqueeze(2)).squeeze(2)    # bring delta to shape (n_shooting_nodes,1,1)
        return  Adelta + self.max_norm_d_M


# Dynamics of a generic system and a generic tube stacked together, without GP. The new state is [x_1,...,x_n, delta]^T
# Note: the forward method describes the time derivative of the system+tube, evaluated at all shooting nodes at once
class SystemAndTubeDynamicsNoGP(torch.nn.Module):
    def __init__(self, system_dynamics, tube_dynamics):
        super().__init__()
        self.system_dynamics = system_dynamics                              
        self.tube_dynamics = tube_dynamics
        self.nx_sys = system_dynamics.nx_sys
        self.ndelta = tube_dynamics.ndelta
        self.nx = system_dynamics.nx_sys + tube_dynamics.ndelta     # total number of states (system + tube)
        self.nu_sys = system_dynamics.nu_sys
        self.nu_extra_opt = 0  # input decision varible is solely u of the system. No extra decision variable such as lambda (RAMPC formulation)
        self.nu = system_dynamics.nu_sys + self.nu_extra_opt     # total number of inputs (system + potential extra decision variables)

    def forward(self, x, u):
        system_output = self.system_dynamics.forward(x[:,:self.nx_sys], u[:,0:self.nu_sys]) # system quantities x and u, for all shooting nodes
        tube_output = self.tube_dynamics.forward(x[:,self.nx_sys].unsqueeze(1))  # delta is a 1d state, we pass a tensor of shape (n_shooting_nodes,1)     
        return torch.cat((system_output, tube_output), dim=1)   # concatenate to get shape (n_shooting_nodes,nx_sys+1)
    
# Dynamics of a generic system, without GP. It can use either the class LinearSystemNoGP or NonlinearSystemNoGP. C
# Note: the forward method describes the time derivative of the system, evaluated at all shooting nodes at once
class SystemDynamicsNoGP(torch.nn.Module):
    def __init__(self, system_dynamics):
        super().__init__()
        self.system_dynamics = system_dynamics                              
        self.nx_sys = system_dynamics.nx_sys
        self.nx = system_dynamics.nx_sys 
        self.nu_sys = system_dynamics.nu_sys
        self.nu_extra_opt = 0  # input decision varible is solely u of the system. No extra decision variable such as lambda (RAMPC formulation)
        self.nu = system_dynamics.nu_sys + self.nu_extra_opt     # total number of inputs (system + potential extra decision variables)

    def forward(self, x, u):
        system_output = self.system_dynamics.forward(x[:,:self.nx_sys], u[:,0:self.nu_sys]) # system quantities x and u, for all shooting nodes   
        return system_output
    

## -------------------------------------------------------------------------------------------------------------------
##
## DYNAMICS WITH GP ESTIMATES
##
## -------------------------------------------------------------------------------------------------------------------

# Dynamics of a linear system, with estimate g_bar from GPs
# Note: the forward method describes the time derivative of the system, evaluated at all shooting nodes at once
class LinearSystemGP(torch.nn.Module):
    def __init__(self, A, B, G):
        super().__init__()
        self.A = A
        self.B = B
        self.G = G
        self.nx_sys = A.shape[1]    # number of system states
        self.nu_sys = B.shape[1]    # number of system inputs

    # Note: x, u and g_bar should be torch tensors of shape (n_shooting_nodes,nx), (n_shooting_nodes,nu), and (n_shooting_nodes,n_gp_outputs)
    def forward(self, x, u, g_bar):
        n_shooting_nodes = x.shape[0]
        A_batch = self.A.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nx)
        B_batch = self.B.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nu)
        G_batch = self.G.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,n_gp_outputs)
        Ax = torch.bmm(A_batch, x.unsqueeze(2)).squeeze(2)  # brings x to shape (n_shooting_nodes,nx,1)
        Bu = torch.bmm(B_batch, u.unsqueeze(2)).squeeze(2)  # brings u to shape (n_shooting_nodes,nu,1)
        Gg_bar = torch.bmm(G_batch, g_bar.unsqueeze(2)).squeeze(2)  #  brings g_bar to shape (n_shooting_nodes,n_gp_outputs,1)

        # shape of output is (n_shouting_nodes,nx)
        if Ax.shape != Bu.shape:
            raise Exception(f"Ax and Bu do not have the same shape.")
        if Ax.shape != Gg_bar.shape:
            raise Exception(f"Ax and Gg_bar do not have the same shape.")
        return  Ax + Bu + Gg_bar


# Dynamics of a quadrotor system, with estimate g_bar from GPs
# Note: the forward method describes the time derivative of the system, evaluated at all shooting nodes at once
class NonlinearSystemGP(torch.nn.Module):
    def __init__(self, B, G, f_fcn):
        super().__init__()
        self.B = B
        self.G = G
        self.f_fcn = f_fcn
        self.nx_sys = B.shape[0]    # number of system states
        self.nu_sys = B.shape[1]    # number of system inputs

    # Note: x and u should be torch tensors of shape (n_shooting_nodes,nx) and (n_shooting_nodes,nu)
    def forward(self, x, u, g_bar):

        n_shooting_nodes = x.shape[0]
        B_batch = self.B.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,nu)
        G_batch = self.G.unsqueeze(0).expand(n_shooting_nodes, -1, -1) # shape (n_shooting_nodes,nx,n_gp_outputs)

        # Note on bbm, batch matrix multiplication: multiplies tensors of size (n_shooting_nodes, n, m), (n_shooting_nodes, m, p) to get (n_shooting_nodes, n, p)
        fx = self.f_fcn(x)
        Bu = torch.bmm(B_batch, u.unsqueeze(2)).squeeze(2)  # brings u to shape (n_shooting_nodes,nu,1), 
        Gg_bar = torch.bmm(G_batch, g_bar.unsqueeze(2)).squeeze(2)  #  brings g_bar to shape (n_shooting_nodes,n_gp_outputs,1)

        # shape of output is (n_shouting_nodes,nx)
        if fx.shape != Bu.shape:
            raise Exception(f"f(x) and Bu do not have the same shape.")
        if fx.shape != Gg_bar.shape:
            raise Exception(f"f(x) and Gg_bar do not have the same shape.")
        
        return  fx + Bu + Gg_bar

# Dynamics of the tube, with bound w_bar from GPs
# Note: the forward method describes the time derivative of the tube, evaluated at all shooting nodes at once
class TubeDynamicsGP(torch.nn.Module):
    def __init__(self, rho, L_G, G_M, E_M, beta):
        super().__init__()
        self.rho = torch.tensor([[rho]])
        self.L_G = L_G
        self.G_M = G_M
        self.E_M = E_M
        self.beta = beta
        self.ndelta = 1     # tube dynamics is always 1-dimensional

    # Note: delta is a torch tensor of shape (n_shooting_nodes,1)
    def forward(self, delta, w_bar): 
        n_shooting_nodes = delta.shape[0]
        A = -(self.rho - self.L_G)
        A_batch = A.unsqueeze(0).expand(n_shooting_nodes, -1, -1)   # shape (n_shooting_nodes,1,1)
        Adelta = torch.bmm(A_batch, delta.unsqueeze(2)).squeeze(2)  # bring delta to shape (n_shooting_nodes,1,1)
        return  Adelta + self.G_M*w_bar + self.E_M


# Dynamics of a generic system and a generic tube stacked together. 
# Note: the forward method describes the time derivative of the system+tube, evaluated at all shooting nodes at once
class SystemAndTubeDynamicsGP(torch.nn.Module):
    def __init__(self, system_dynamics, tube_dynamics, estimate_model):
        super().__init__()
        self.system_dynamics = system_dynamics
        self.tube_dynamics = tube_dynamics
        self.estimate_model = estimate_model
        self.nx_sys = system_dynamics.nx_sys
        self.ndelta = tube_dynamics.ndelta
        self.nx = system_dynamics.nx_sys + tube_dynamics.ndelta
        self.nu_sys = system_dynamics.nu_sys
        self.nu_extra_opt = 0   # no extra input in the basic formulation, we consider prediction with a single GP model (no need for lambda)
        self.nu = system_dynamics.nu_sys + self.nu_extra_opt     # total number of inputs (system + potential extra decision variables)
        self.g_sys_state_idx = estimate_model.g_sys_state_idx

    def forward(self, x, u):
        g_bar, w_bar = self.estimate_model(x[:,self.g_sys_state_idx])    # return tensors of shape (n_shooting_nodes, n_gp_outputs=1). x-index "self.g_sys_state_idx" specific to our system's GP/function g. Example: 2,3,4 for quadrotor
        system_output = self.system_dynamics.forward(x[:,:self.nx_sys], u[:,0:self.nu_sys], g_bar.unsqueeze(1)) # system quantities x and u, for all shooting nodes
        tube_output = self.tube_dynamics.forward(x[:,self.nx_sys].unsqueeze(1), w_bar.unsqueeze(1))  # delta is a 1d state, we pass a tensor of shape (n_shooting_nodes,1)     
        return torch.cat((system_output, tube_output), dim=1)   # concatenate to get shape (n_shooting_nodes,nx_sys+1)
    

# Dynamics of a generic system and a generic tube stacked together for the multi lambda formulation. The new state is [x_1,x_2,..., delta]^T.
# TODO: finish to implement/double-check
class SystemAndTubeDynamicsGPMultiLambda(torch.nn.Module):
    def __init__(self, system_dynamics, tube_dynamics, estimate_model):
        super().__init__()
        self.system_dynamics = system_dynamics
        self.tube_dynamics = tube_dynamics
        self.estimate_model = estimate_model
        self.nx_sys = system_dynamics.nx_sys
        self.ndelta = tube_dynamics.ndelta
        self.nx = system_dynamics.nx_sys + tube_dynamics.ndelta
        self.nu_sys = system_dynamics.nu_sys
        self.nu_extra_opt = estimate_model.n_gp_in_batch        # additional input/optimisation variable lambda
        self.nu = system_dynamics.nu_sys + self.nu_extra_opt    # total number of inputs (system + potential extra decision variables)
        self.g_sys_state_idx = estimate_model.g_sys_state_idx

    def forward(self, x, u):
        g_bar, w_bar = self.estimate_model(x[:,self.g_sys_state_idx], u[:,self.nu_sys:])    # return tensors of shape (n_shooting_nodes, n_gp_outputs=1). x-index specific to our system's GP
        # print(w_bar)
        # w_bar = 0.04*torch.ones(w_bar.shape[0])
        system_output = self.system_dynamics.forward(x[:,:self.nx_sys], u[:,0:self.nu_sys], g_bar.unsqueeze(1)) # system quantities x and u, for all shooting nodes
        tube_output = self.tube_dynamics.forward(x[:,self.nx_sys].unsqueeze(1), w_bar.unsqueeze(1))  # delta is a 1d state, we pass a tensor of shape (n_shooting_nodes,1)  
        # print('current value of delta')
        # print(x[0:2,self.nx_sys])  
        return torch.cat((system_output, tube_output), dim=1)   # concatenate to get shape (n_shooting_nodes,nx_sys+1)


    

## -------------------------------------------------------------------------------------------------------------------
##
## DYNAMICS OF REAL SYSTEMS
##
## -------------------------------------------------------------------------------------------------------------------

# Real linear system dynamics
# Note: dynamics defined for a single shooting node, as it is used for single step simulation only
class LinearSystemReal(torch.nn.Module):
    def __init__(self, A, B, G, g_fcn, E=None, d_bounds=None):
        super().__init__()
        self.A = A
        self.B = B
        self.G = G
        self.g_fcn = g_fcn
        self.E = E
        self.d_bounds = d_bounds    # torch tensor of shape (n_d,2) with min and max bound of noise
        self.nx_sys = B.shape[0]    # number of system's state
        self.nu_sys = B.shape[1]    # number of system's input

    def forward(self, x, u):
        Ax = torch.matmul(self.A, x)
        Bu = torch.matmul(self.B, u) 
        g = self.g_fcn(x) 
        Gg = torch.matmul(self.G,g)
        if self.d_bounds is not None: # case with noise 
            lower_bounds = self.d_bounds[:, 0]
            upper_bounds = self.d_bounds[:, 1]
            d_vector = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(self.d_bounds.shape[0])
            if d_vector.shape[0] == 1:
                d_vector = d_vector.unsqueeze(0)
            Ed = torch.matmul(self.E,d_vector)

        if Ax.shape != Bu.shape:
            raise Exception(f"Ax and Bu do not have the same shape.")
        if Ax.shape != Gg.shape:
            raise Exception(f"Ax and Gg do not have the same shape.")
        
        if self.d_bounds is not None:
            if Ax.shape != Ed.shape:
                raise Exception(f"Ax and Ed do not have the same shape.")
            x_next = Ax + Bu + Gg + Ed
        else:
            x_next = Ax + Bu + Gg 

        return  x_next


# Dynamics of the real quadrotor system
# Note: dynamics defined for a single shooting node, as it is used for single step simulation only
class NonlinearSystemReal(torch.nn.Module):
    def __init__(self, B, G, f_fcn, g_fcn, E=None, d_bounds=None):
        super().__init__()
        self.B = B
        self.G = G
        self.f_fcn = f_fcn
        self.g_fcn = g_fcn
        self.E = E
        self.d_bounds = d_bounds    # torch tensor of shape (n_d,2) with min and max bound of noise
        self.nx_sys = B.shape[0]    # number of system states
        self.nu_sys = B.shape[1]    # number of system inputs
    
    def forward(self, x, u):

        fx = self.f_fcn(x)
        gx = self.g_fcn(x)
        Gg = torch.matmul(self.G, gx)
        if Gg.dim() == 1:     # g is scalar
            Gg = Gg.unsqueeze(1)    # bring to shape (nx,1)

        if self.d_bounds is not None: # case with noise 
            lower_bounds = self.d_bounds[:, 0]
            upper_bounds = self.d_bounds[:, 1]
            d_vector = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(self.d_bounds.shape[0])
            if d_vector.shape[0] == 1:
                d_vector = d_vector.unsqueeze(0)
            Ed = torch.matmul(self.E,d_vector)

        Bu = torch.matmul(self.B, u)

        if fx.shape != Bu.shape:
            raise Exception(f"f(x) and Bu do not have the same shape.")
        if fx.shape != Gg.shape:
            raise Exception(f"f(x) and Gg do not have the same shape.")
        
        if self.d_bounds is not None:
            if fx.shape != Ed.shape:
                raise Exception(f"f(x) and Ed do not have the same shape.")
            x_next = fx + Bu + Gg + Ed
        else:
            x_next = fx + Bu + Gg 
        
        return  x_next

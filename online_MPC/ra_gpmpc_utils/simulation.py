import torch
import numpy as np

# Compute approximation of the geodesic: straight_line between x and y
def compute_geodesic_approx(x,z,s):
    gamma = z + s*(x-z)
    gamma_s = x-z
    return gamma, gamma_s

# Compute the input according to the feedback law kappa. Integrate along the geodesic using finite discretization
def kappa(x,z,v,geod_fcn,Y_fcn,W_fcn,n_discretization_intervals):
    s_sequence = np.linspace(0,1,num=n_discretization_intervals+1)
    ds = 1/n_discretization_intervals
    u = v
    for i in range(n_discretization_intervals):
        s_val = s_sequence[i]
        gamma_at_s, gamma_s_at_s = geod_fcn(x,z,s_val)  # tensors
        Y_at_gamma = torch.tensor(Y_fcn(gamma_at_s[2].detach().numpy(), gamma_at_s[3].detach().numpy()))
        W_at_gamma = torch.tensor(W_fcn(gamma_at_s[2].detach().numpy(), gamma_at_s[3].detach().numpy()))
        K_at_gamma = torch.matmul(Y_at_gamma, torch.inverse(W_at_gamma)).float()    # convert from float 64 to float 32, which is the standard torch dtype
        u += torch.matmul(K_at_gamma, gamma_s_at_s)*ds
    return u


# Simulate the nominal system over the discretization time dt, based on constant nominal input v, RMPC estimate model (no lambda)
def simulate_nom_system_RK4_1_sn_RMPC(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, dt):
    g_bar_k1, _ = estimate_model(z[:,g_sys_state_idx])
    k1 = nominal_sys_ode(z, v, g_bar_k1.unsqueeze(1))

    g_bar_k2, _ = estimate_model(z[:,g_sys_state_idx] + 0.5*dt*k1[:,g_sys_state_idx])
    k2 = nominal_sys_ode(z + 0.5*dt*k1, v, g_bar_k2.unsqueeze(1))

    g_bar_k3, _ = estimate_model(z[:,g_sys_state_idx] + 0.5*dt*k2[:,g_sys_state_idx])
    k3 = nominal_sys_ode(z + 0.5*dt*k2, v, g_bar_k3.unsqueeze(1))

    g_bar_k4, _ = estimate_model(z[:,g_sys_state_idx] + dt*k3[:,g_sys_state_idx])
    k4 = nominal_sys_ode(z + dt*k3, v, g_bar_k4.unsqueeze(1))
    
    z_next = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return z_next

# Simulate the nominal system over the discretization time dt, based on constant nominal input v. RAMPC estimate model (with lambda)
def simulate_nom_system_RK4_1_sn(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, lambda_vec, dt):
    g_bar_k1, _ = estimate_model(z[:,g_sys_state_idx], lambda_vec)
    k1 = nominal_sys_ode(z, v, g_bar_k1.unsqueeze(1))

    g_bar_k2, _ = estimate_model(z[:,g_sys_state_idx] + 0.5*dt*k1[:,g_sys_state_idx], lambda_vec)
    k2 = nominal_sys_ode(z + 0.5*dt*k1, v, g_bar_k2.unsqueeze(1))

    g_bar_k3, _ = estimate_model(z[:,g_sys_state_idx] + 0.5*dt*k2[:,g_sys_state_idx], lambda_vec)
    k3 = nominal_sys_ode(z + 0.5*dt*k2, v, g_bar_k3.unsqueeze(1))

    g_bar_k4, _ = estimate_model(z[:,g_sys_state_idx] + dt*k3[:,g_sys_state_idx], lambda_vec)
    k4 = nominal_sys_ode(z + dt*k3, v, g_bar_k4.unsqueeze(1))
    
    z_next = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return z_next


# %%
# Simulate the real system over the discretization time dt, based on a nominal input v. Constant matrix K, RMPC formulation (no lambda)
# Note: the nominal system is simulated as well to use feedback control over the discretization time
def simulate_real_system_RK4_1_sn_const_K_RMPC(real_sys_ode, nominal_sys_ode, estimate_model, g_sys_state_idx, x, z, v, K, dt):

    # simulate nominal system to apply feedback control
    z_k2_k3 = simulate_nom_system_RK4_1_sn_RMPC(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, 0.5*dt)
    z_k4 = simulate_nom_system_RK4_1_sn_RMPC(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, dt)

    # change shape of x, z, and v: (1,nx) -> (nx,1), (1,nx) --> (nx,1), (1,nu) --> (nu,1)
    # note: done like this as we have a single shooting node here
    x = torch.transpose(x,0,1)
    z = torch.transpose(z,0,1)
    v = torch.transpose(v,0,1)
    z_k2_k3 = torch.transpose(z_k2_k3,0,1)
    z_k4 = torch.transpose(z_k4,0,1)   

    # k1
    u_k1 = v + torch.matmul(K, x - z)
    k1 = real_sys_ode(x, u_k1)

    # k2 
    u_k2 = v + torch.matmul(K, x + 0.5*dt*k1 - z_k2_k3)
    k2 = real_sys_ode(x + 0.5*dt*k1, u_k2)

    # k3
    u_k3 = v + torch.matmul(K, x + 0.5*dt*k2 - z_k2_k3)
    k3 = real_sys_ode(x + 0.5*dt*k2, u_k3)

    # k4
    u_k4 = v + torch.matmul(K, x + dt*k3 - z_k4)
    k4 = real_sys_ode(x + dt*k3, u_k4)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
    
# Simulate the real system over the discretization time dt, based on a nominal input v. Constant matrix K, RAMPC formulation (with lambdas)
# Note: the nominal system is simulated as well to use feedback control over the discretization time
def simulate_real_system_RK4_1_sn_const_K(real_sys_ode, nominal_sys_ode, estimate_model, g_sys_state_idx, x, z, v, lambda_vec, K, dt):

    # simulate nominal system to apply feedback control
    z_k2_k3 = simulate_nom_system_RK4_1_sn(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, lambda_vec, 0.5*dt)
    z_k4 = simulate_nom_system_RK4_1_sn(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, lambda_vec, dt)

    # change shape of x, z, and v: (1,nx) -> (nx,1), (1,nx) --> (nx,1), (1,nu) --> (nu,1)
    # note: done like this as we have a single shooting node here
    x = torch.transpose(x,0,1)
    z = torch.transpose(z,0,1)
    v = torch.transpose(v,0,1)
    z_k2_k3 = torch.transpose(z_k2_k3,0,1)
    z_k4 = torch.transpose(z_k4,0,1)
    
    # k1
    u_k1 = v + torch.matmul(K, x - z)
    k1 = real_sys_ode(x, u_k1)

    # k2 
    u_k2 = v + torch.matmul(K, x + 0.5*dt*k1 - z_k2_k3)
    k2 = real_sys_ode(x + 0.5*dt*k1, u_k2)

    # k3
    u_k3 = v + torch.matmul(K, x + 0.5*dt*k2 - z_k2_k3)
    k3 = real_sys_ode(x + 0.5*dt*k2, u_k3)

    # k4
    u_k4 = v + torch.matmul(K, x + dt*k3 - z_k4)
    k4 = real_sys_ode(x + dt*k3, u_k4)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
    

# %%    
# Simulate the real system over the discretization time dt, based on a nominal input v. Feedback K is a function, RMPC formulation (no lambdas)
# Note: the nominal system is simulated as well to use feedback control over the discretization time
def simulate_real_system_RK4_1_sn_RMPC(real_sys_ode, nominal_sys_ode, estimate_model, g_sys_state_idx, x, z, v, Y_fcn, W_fcn, dt):

    # simulate nominal system to apply feedback control
    z_k2_k3 = simulate_nom_system_RK4_1_sn_RMPC(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, 0.5*dt)
    z_k4 = simulate_nom_system_RK4_1_sn_RMPC(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, dt)

    # change shape of x, z, and v: (1,nx) -> (nx,1), (1,nx) --> (nx,1), (1,nu) --> (nu,1)
    # note: done like this as we have a single shooting node here
    x = torch.transpose(x,0,1)
    z = torch.transpose(z,0,1)
    v = torch.transpose(v,0,1)
    z_k2_k3 = torch.transpose(z_k2_k3,0,1)
    z_k4 = torch.transpose(z_k4,0,1)
    
    n_discretization_pts = 10

    # k1, u_k1 of shape (n_input,1)
    u_k1 = kappa(x, z, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts)  # TODO: the geodesic could be computed by solving an optimisation problem instead of being approximated
    k1 = real_sys_ode(x, u_k1)

    # k2, u_k2 of shape (n_input,1)
    u_k2 = kappa(x + 0.5*dt*k1, z_k2_k3, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts)   
    k2 = real_sys_ode(x + 0.5*dt*k1, u_k2)

    # k3, u_k3 of shape (n_input,1)
    u_k3 = kappa(x + 0.5*dt*k2, z_k2_k3, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts) 
    k3 = real_sys_ode(x + 0.5*dt*k2, u_k3)

    # k4, u_k4 of shape (n_input,1)
    u_k4 = kappa(x + dt*k3, z_k4, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts) 
    k4 = real_sys_ode(x + dt*k3, u_k4)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
    
# Simulate the real system over the discretization time dt, based on a nominal input v. Feedback K is a function, RAMPC formulation (with lambdas)
# Note: the nominal system is simulated as well to use feedback control over the discretization time
def simulate_real_system_RK4_1_sn(real_sys_ode, nominal_sys_ode, estimate_model, g_sys_state_idx, x, z, v, lambda_vec, Y_fcn, W_fcn, dt):

    # simulate nominal system to apply feedback control
    z_k2_k3 = simulate_nom_system_RK4_1_sn(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, lambda_vec, 0.5*dt)
    z_k4 = simulate_nom_system_RK4_1_sn(nominal_sys_ode, estimate_model, g_sys_state_idx, z, v, lambda_vec, dt)

    # change shape of x, z, and v: (1,nx) -> (nx,1), (1,nx) --> (nx,1), (1,nu) --> (nu,1)
    # note: done like this as we have a single shooting node here
    x = torch.transpose(x,0,1)
    z = torch.transpose(z,0,1)
    v = torch.transpose(v,0,1)
    z_k2_k3 = torch.transpose(z_k2_k3,0,1)
    z_k4 = torch.transpose(z_k4,0,1)
    
    n_discretization_pts = 10

    # k1
    u_k1 = kappa(x, z, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts)
    k1 = real_sys_ode(x, u_k1)

    # k2 
    u_k2 = kappa(x + 0.5*dt*k1, z_k2_k3, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts)   
    k2 = real_sys_ode(x + 0.5*dt*k1, u_k2)

    # k3
    u_k3 = kappa(x + 0.5*dt*k2, z_k2_k3, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts) 
    k3 = real_sys_ode(x + 0.5*dt*k2, u_k3)

    # k4
    u_k4 = kappa(x + dt*k3, z_k4, v, compute_geodesic_approx, Y_fcn, W_fcn, n_discretization_pts) 
    k4 = real_sys_ode(x + dt*k3, u_k4)

    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
    

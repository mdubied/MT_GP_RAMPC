import sys, os
import numpy as np
import gpytorch
import torch
import random

def generate_train_inputs_zoro(zoro_solver, x0_nom, N_sim_per_x0, N_x0, 
    random_seed=None, 
    x0_rand_scale=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    nx = zoro_solver.nx
    nu = zoro_solver.nu
    N = zoro_solver.N

    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0*N_sim_per_x0*N, nx+nu))

    i_fac = N_sim_per_x0 * N
    j_fac = N
    for i in range(N_x0):
        x0_arr[i,:] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)

        zoro_solver.ocp_solver.set(0, "lbx", x0_arr[i,:])
        zoro_solver.ocp_solver.set(0, "ubx", x0_arr[i,:])
        zoro_solver.solve()

        X,U,P = zoro_solver.get_solution()
        
        # store training points
        for j in range(N_sim_per_x0):
            for k in range(N):
                ijk = i*i_fac+j*j_fac+k
                X_inp[ijk,:] = np.hstack((
                    X[k,:],
                    U[k,:]
                ))
    
    return X_inp, x0_arr

def generate_train_inputs_acados(ocp_solver, x0_nom, N_sim_per_x0, N_x0, 
    random_seed=None, 
    x0_rand_scale=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    N = ocp_solver.acados_ocp.dims.N

    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0*N_sim_per_x0*N, nx+nu))

    i_fac = N_sim_per_x0 * N
    j_fac = N
    for i in range(N_x0):
        x0_arr[i,:] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)

        ocp_solver.set(0, "lbx", x0_arr[i,:])
        ocp_solver.set(0, "ubx", x0_arr[i,:])
        ocp_solver.solve()
        
        # store training points
        for j in range(N_sim_per_x0):
            for k in range(N):
                ijk = i*i_fac+j*j_fac+k
                X_inp[ijk,:] = np.hstack((
                    ocp_solver.get(k, "x"),
                    ocp_solver.get(k, "u")
                ))
    
    return X_inp, x0_arr


def generate_train_data_acados(acados_ocp_solver, 
    integrator_nom,
    integrator_sim,
    x0_nom,
    Sigma_W, 
    N_sim_per_x0, 
    N_sim,
    B=None,
    N_x0=1, 
    random_seed=None, 
    x0_rand_scale=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    if B is None:
        B = np.eye(nw)
    B_inv = np.linalg.pinv(B)

    nx = acados_ocp_solver.acados_ocp.model.x.size()[0]
    nu = acados_ocp_solver.acados_ocp.model.u.size()[0]
    nw = Sigma_W.shape[0]

    B_inv = np.linalg.pinv(B)
    x0_arr = np.zeros((N_x0, nx))
    X_inp = np.zeros((N_x0*N_sim_per_x0*N_sim, nx+nu))
    Y_out = np.zeros((N_x0*N_sim_per_x0*N_sim, nw))

    i_fac = N_sim_per_x0 * N_sim
    j_fac = N_sim
    for i in range(N_x0):
        ijk = i*i_fac
        x0_arr[i,:] = x0_nom + x0_rand_scale * (2 * np.random.rand(nx) - 1)
        xcurrent = x0_arr[i,:]
        for j in range(N_sim_per_x0):
            for k in range(N_sim):
                acados_ocp_solver.set(0, "lbx", xcurrent)
                acados_ocp_solver.set(0, "ubx", xcurrent)
                acados_ocp_solver.solve()

                u = acados_ocp_solver.get(0, "u")

                # integrate nominal model
                integrator_nom.set("x", xcurrent)
                integrator_nom.set("u", u)
                integrator_nom.solve()
                xnom = integrator_nom.get("x")

                # integrate real model
                integrator_sim.set("x", xcurrent)
                integrator_sim.set("u", u)
                integrator_sim.solve()
                xcurrent = integrator_sim.get("x")

                # difference
                w = np.random.multivariate_normal(np.zeros((nw,)), Sigma_W)

                # store training points
                ijk = i*i_fac+j*j_fac+k
                X_inp[ijk,:] = np.hstack((
                    xcurrent,
                    u
                ))

                Y_out[ijk,:] = B_inv @ (xcurrent - xnom) + w 
    
    return X_inp, Y_out

def generate_train_outputs_at_inputs(X_inp, integrator_nom, integrator_sim, Sigma_W, B=None):
    
    nx = integrator_nom.acados_sim.model.x.size()[0]
    nu = integrator_nom.acados_sim.model.u.size()[0]
    nw = Sigma_W.shape[0]

    if B is None:
        B = np.eye(nw)
    B_inv = np.linalg.pinv(B)

    n_train = X_inp.shape[0]
    Y_out = np.zeros((n_train, nw))
    for i in range(n_train):
        integrator_sim.set("x", X_inp[i,0:nx])
        integrator_sim.set("u", X_inp[i,nx:nx+nu])
        integrator_sim.solve()

        integrator_nom.set("x", X_inp[i,0:nx])
        integrator_nom.set("u", X_inp[i,nx:nx+nu])
        integrator_nom.solve()        

        w = np.random.multivariate_normal(np.zeros((nw,)), Sigma_W)
        
        Y_out[i,:] = B_inv @ (integrator_sim.get("x") - integrator_nom.get("x")) + w  
    return Y_out

def train_gp_model(gp_model, torch_seed = None, training_iterations = 200):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    likelihood = gp_model.likelihood
    train_x = gp_model.train_inputs[0]
    train_y = gp_model.train_targets

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gp_model.parameters()), 
        lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = likelihood(gp_model(train_x))
        loss = -mll(output, train_y.reshape((train_y.numel(),)))
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    return gp_model, likelihood

def set_gp_param_value(gp_model, name: str, value: torch.Tensor):
    constraint = gp_model.constraint_for_parameter_name(name)
    parameter = gp_model.get_parameter(name)
    state_dict = gp_model.state_dict()

    # transform value
    if constraint is not None:
        value_transform = constraint.inverse_transform(value)
    else:
        value_transform = value

    state_dict[name] = value_transform
    gp_model.load_state_dict(state_dict)
    
    return parameter

def get_gp_param_value(gp_model, name: str):
    constraint = gp_model.constraint_for_parameter_name(name)
    parameter = gp_model.get_parameter(name)
    state_dict = gp_model.state_dict()
    value_transform = state_dict[name]

    # transform value
    if constraint is not None:
        value = constraint.transform(value_transform)
    else:
        value = value_transform
    return value

def get_gp_param_names(gp_model):
    names = []
    for name, parameter in gp_model.named_parameters():
        names += [name]
    return names

def get_gp_param_names_values(gp_model):
    names = get_gp_param_names(gp_model)
    values = []
    for name in names:
        values += [get_gp_param_value(gp_model, name)]
    return zip(names, values)

def get_prior_covariance(gp_model):

    dim = gp_model.train_inputs[0].shape[1]

    # cuda check
    if gp_model.train_inputs[0].device.type == "cuda":
        to_numpy = lambda T: T.cpu().numpy()
        y_test = torch.Tensor(np.ones((1,dim))).cuda()
    else:
        to_numpy = lambda T: T.numpy()
        y_test = torch.Tensor(np.ones((1,dim)))


    gp_model.eval()
    # This only works for batched shape now
    # TODO: Make this work again for general multitask kernel
    prior_covar = np.diag(
        to_numpy(gp_model.covar_module(y_test))
        .squeeze()
    )
    
    return prior_covar

# FROM HERE ONWARD: Functions added specifically for nonparamatric learning MPC
# Author: Mathieu Dubied (mdubied@ethz.ch)
# Date: 17/06/2024

# Generate training positiona X_train_offline as a grid over specified ranges for \phi, v1 and v2
def generate_training_positions_as_grid(phi_range, v1_range, v2_range, num_points):
    # Generate grid points within the specified ranges for each variable
    phi_points = torch.linspace(phi_range[0], phi_range[1], num_points)
    v1_points = torch.linspace(v1_range[0], v1_range[1], num_points)
    v2_points = torch.linspace(v2_range[0], v2_range[1], num_points)
    
    # Generate all combinations of grid points
    phi_grid, v1_grid, v2_grid = torch.meshgrid(phi_points, v1_points, v2_points)
    
    # Flatten the grid points and stack them into a single matrix
    X_train_offline = torch.stack([phi_grid.flatten(), v1_grid.flatten(), v2_grid.flatten()], dim=1)
    
    return X_train_offline

# Generate values of Y_train, a tensor of size (n_data, 1).
# X_train is a tensor of size (n_data,n_GP_inputs). 
# g_eval consider a subset of the state variables as argument, namely only the state used as GP_inputs
# TODO: extend/test the function for multi-outputs function g
def generate_g_outputs_at_training_inputs(g_eval, X_train, noise_size):

    n_train = X_train.shape[0]
    Y_train = torch.zeros(n_train)

    for i in range(0,n_train):
        g_value = g_eval(X_train[i,:])  # g_eval evaluate each observation of size n_GP_inputs separately
        noise_value = (torch.rand(1)-0.5)*noise_size
        Y_train[i] = g_value + noise_value

    return Y_train

# Train the GPs as a batch with output measurement that are NaN
def train_GP_models_as_batch_with_nan(gp_models_as_batch, likelihood, X_train_batch, Y_train_batch, training_iterations = 200):
    # 'mask' option ok if all GP along the batch dimension contain the same training data
    with gpytorch.settings.observation_nan_policy("mask"):
        
        # Training mode
        gp_models_as_batch.train()
        likelihood.train()

        optimizer = torch.optim.Adam(gp_models_as_batch.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_models_as_batch)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = gp_models_as_batch(X_train_batch)
            loss = -mll(output, Y_train_batch).sum()  # Summing the losses across the batch dimensions
            loss.backward()
            print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item():.3f}')
            optimizer.step()

    return gp_models_as_batch, likelihood
    

# Function to set hyperparameters of all batch dimensions to be the same as the first batch dimension
def set_hyperparameters_to_first_dimension(model, likelihood):
    with torch.no_grad():
        # Mean parameter
        model.mean_module.raw_constant.copy_(model.mean_module.raw_constant[0].expand_as(model.mean_module.raw_constant))

        # Covariance parameters
        model.covar_module.raw_outputscale.copy_(model.covar_module.raw_outputscale[0].expand_as(model.covar_module.raw_outputscale))
        model.covar_module.base_kernel.raw_lengthscale.copy_(model.covar_module.base_kernel.raw_lengthscale[0].expand_as(model.covar_module.base_kernel.raw_lengthscale))

        # Noise parameter
        likelihood.noise_covar.raw_noise.copy_(likelihood.noise_covar.raw_noise[0].expand_as(likelihood.noise_covar.raw_noise))
    
    return model, likelihood

# Print hyperparameters of the GP models of the batch
def print_hyperparameters_of_batch(gp_models_as_batch, likelihood, show_raw_params = False):
    # Print hyperparameters
    print("Mean constant:", gp_models_as_batch.mean_module.constant)
    print("Output scale:", gp_models_as_batch.covar_module.outputscale)
    print("Lengthscale:", gp_models_as_batch.covar_module.base_kernel.lengthscale)
    print("Noise:", likelihood.noise_covar.noise)

    # Print raw hyperparameters if flag set to true
    if show_raw_params == True:
        for name, param in gp_models_as_batch.named_parameters():
            print(f"{name}: {param.data}")
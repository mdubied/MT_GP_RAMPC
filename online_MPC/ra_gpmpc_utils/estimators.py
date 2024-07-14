import torch
import gpytorch

# Provide the GP estimate and bound needed for the system and tube dynamics for the RMPC case (single GP model)
class EstimateAndBoundSingleGP(torch.nn.Module):
    def __init__(self, gp_model, gp_beta, g_sys_state_idx):
        super().__init__()
        self.gp_model = gp_model
        self.gp_beta = gp_beta
        self.g_sys_state_idx = g_sys_state_idx

    # evaluate mean \mu and standard deviation \sigma of the GP model (at x)
    def evaluate_mu_sigma(self,x):
        with gpytorch.settings.observation_nan_policy("fill"):
            prediction = self.gp_model(x)
            
        mu = prediction.mean[0,:]   # shape is (n_gp_models_in_batch,n_shouting_nodes), we select the first GP model for this formulation
        sigma = torch.sqrt(prediction.variance[0,:])
        return mu, sigma

    # forward method: returns g_bar = mu and w_bar = \beta*\sigma evaluated at x
    def forward(self, x):
        mu, sigma = self.evaluate_mu_sigma(x)
        g_bar = mu
        w_bar = self.gp_beta*sigma
        return g_bar, w_bar
 
 # Provide the GP estimate and bound needed for the system and tube dynamics, using  additional decision variables named \lambda
class EstimateAndBoundMultiLambda(torch.nn.Module):
    def __init__(self, gp_models_as_batch, gp_betas, active_gp_idx, g_sys_state_idx):
        super().__init__()
        self.gp_models_as_batch = gp_models_as_batch    # GP models in a batch
        self.n_gp_in_batch = gp_models_as_batch.mean_module.batch_shape[0]  # total number of GP in the batch     
        self.gp_betas = gp_betas    # bounds from literature (|mu_i(x)-g(x)\leq \beta_i \sigma_i(x)). List. Will be converted to a torch tensor for mathematical operation
        self.active_gp_idx = active_gp_idx # list of the GP models in the batch that are active
        self.n_active_gp = len(active_gp_idx)   # number of active GP models in our collection/list
        self.g_sys_state_idx = g_sys_state_idx
        self.last_w_bar_gp_idx = 0  # store the last GP model indexes used in the computation of w_bar (see forward method) 

    # evaluate mean \mu and standard deviation \sigma of all (active) GP in GP models' collection at a given x
    def evaluate_mu_sigma(self,x):
        with gpytorch.settings.observation_nan_policy("fill"):
            prediction = self.gp_models_as_batch(x)
            
        mu = prediction.mean[:,:]   # shape is (n_gp_models_in_batch,n_shouting_nodes)
        sigma = torch.sqrt(prediction.variance[:,:])
        return mu, sigma
    
    def compute_g_bar(self, lmbda_all, mu_all):
        
        product = lmbda_all[:, self.active_gp_idx]* mu_all[:, self.active_gp_idx]    # element-wise multiplication
        g_bar = product.sum(dim=1)  # weigthed sum at each shooting nodes. shape is (n_shooting_nodes)
        return g_bar
    
    # compute \bar{w}(x) = max{min_i{\mu_i(x) + \beta_i\sigma_i(x) - \bar{g}}, \bar{g} - max_i{\mu_i(x) - \beta_i\sigma(x)}}
    # also return the index of the GP model that leads to the \bar{w}(x) value
    def compute_w_bar(self, mu_list, sigma_list, g_bar):
        # get the active GP indices
        active_gp_idx = self.active_gp_idx #= torch.tensor(self.active_gp_idx)
        
        # select the active mu and sigma based on active_gp_idx
        mu_active = mu_list[:, active_gp_idx]  # shape: (n_shooting_nodes, n_active_gp)
        sigma_active = sigma_list[:, active_gp_idx]  # shape: (n_shooting_nodes, n_active_gp)
        gp_betas_active = torch.tensor([self.gp_betas[i] for i in active_gp_idx])  # shape: torch.Size([n_active_gp])

        # compute upper and lower bounds
        upper_bound_list = mu_active + gp_betas_active * sigma_active   # element-wise multiplication
        lower_bound_list = mu_active - gp_betas_active * sigma_active   # element-wise multiplication

        # compute min and max bounds
        min_upper_bound, min_upper_indices = upper_bound_list.min(dim=1)  # shape: (n_shooting_nodes,)
        max_lower_bound, max_lower_indices = lower_bound_list.max(dim=1)  # shape: (n_shooting_nodes,)

        min_upper_bound_g_bar = min_upper_bound - g_bar  # shape: (n_shooting_nodes,)
        g_bar_max_lower_bound = g_bar - max_lower_bound  # shape: (n_shooting_nodes,)

        # select corresponding indices from active_gp_idx
        index_min_upper_bound = [active_gp_idx[i] for i in min_upper_indices.tolist()]  # shape: (n_shooting_nodes,), a list
        index_max_lower_bound = [active_gp_idx[i] for i in max_lower_indices.tolist()]  # shape: (n_shooting_nodes,), a list
        index_min_upper_bound = torch.tensor(index_min_upper_bound)  # shape: (n_shooting_nodes,), convert to tensor
        index_max_lower_bound = torch.tensor(index_max_lower_bound)  # shape: (n_shooting_nodes,), convert to tensor

        # check for intersection and compute the final result
        no_intersection = max_lower_bound > min_upper_bound
        if no_intersection.any():
            raise Exception("No intersection between the beta*sigma upper and lower bounds.")

        condition = min_upper_bound_g_bar > g_bar_max_lower_bound 
        result = torch.where(condition, min_upper_bound_g_bar, g_bar_max_lower_bound)   # we select max(min_upper_bound-g_bar, g_Bar - max_lower_bound)
        result_indices = torch.where(condition, index_min_upper_bound, index_max_lower_bound)

        if (result < 0).any():
            raise Exception("Bound w(x) computed as smaller than 0.")
        return result, result_indices


    # forward method: returns g_bar and w_bar evaluated at x, x having the shape (n_shooting_nodes,n_gp_input)
    # lmbda_list has the shape (n_shooting_nodes,n_lambda_per_stage)
    def forward(self, x, lmbda_list):
        mu_all_gp_all_shooting_nodes,  sigma_all_gp_all_shooting_nodes = self.evaluate_mu_sigma(x)
        mu_all_gp_all_shooting_nodes = torch.transpose(mu_all_gp_all_shooting_nodes,0,1)
        sigma_all_gp_all_shooting_nodes = torch.transpose(sigma_all_gp_all_shooting_nodes,0,1)
        g_bar = self.compute_g_bar(mu_all_gp_all_shooting_nodes, lmbda_list)
        w_bar, w_bar_gp_idx = self.compute_w_bar(mu_all_gp_all_shooting_nodes, sigma_all_gp_all_shooting_nodes, g_bar)
        self.last_w_bar_gp_idx = w_bar_gp_idx.tolist()
        # print(w_bar)
        return g_bar, w_bar
    
import sys, os
import numpy as np
import gpytorch
import torch
import math
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import List

@dataclass
class GPlotData:
    x_path: np.ndarray = None,
    mean_on_path: np.ndarray = None,
    stddev_on_path: np.ndarray = None,
    conf_lower_on_path: np.ndarray = None,
    conf_upper_on_path: np.ndarray = None,
    sample_on_path: np.ndarray = None,
    train_x: np.ndarray = None,
    train_y: np.ndarray = None

def gp_data_from_model_and_path(gp_model, likelihood, x_path,
    num_samples = 0,
    use_likelihood = False
):
    # GP data
    if gp_model.train_inputs[0].device.type == "cuda":
        # train_x = gp_model.train_inputs[0].cpu().numpy()
        # train_y = gp_model.train_targets.cpu().numpy()
        x_path_tensor = torch.Tensor(x_path).cuda()
        to_numpy = lambda T: T.cpu().numpy()
    else:
        # train_x = gp_model.train_inputs[0].numpy()
        # train_y = gp_model.train_targets.numpy()
        x_path_tensor = torch.Tensor(x_path)
        to_numpy = lambda T: T.numpy()

    train_x = to_numpy(gp_model.train_inputs[0])
    train_y = to_numpy(gp_model.train_targets)

    # dimensions
    num_points = x_path.shape[0]
    nout = train_y.shape[1]

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if use_likelihood:
            predictions = likelihood(gp_model(x_path_tensor)) # predictions with noise
        else:
            predictions = gp_model(x_path_tensor) # only model (we want to find true function)
        
        mean_on_path = predictions.mean
        stddev_on_path = predictions.stddev
        conf_lower_on_path, conf_upper_on_path = predictions.confidence_region()

        # reshape
        mean_on_path = to_numpy(mean_on_path).reshape((num_points,nout))
        stddev_on_path = to_numpy(stddev_on_path).reshape((num_points,nout))
        conf_lower_on_path = to_numpy(conf_lower_on_path).reshape((num_points,nout))
        conf_upper_on_path = to_numpy(conf_upper_on_path).reshape((num_points,nout))

    # draw function samples
    sample_on_path = np.zeros((num_points,nout,num_samples))
    for j in range(0,num_samples):
        sample_on_path[:,:,j] = to_numpy(predictions.sample()).reshape((num_points,nout))

    return GPlotData(
        x_path,
        mean_on_path,
        stddev_on_path,
        conf_lower_on_path,
        conf_upper_on_path,
        sample_on_path,
        train_x,
        train_y
    )

def gp_derivative_data_from_model_and_path(gp_model, likelihood, x_path,
    num_samples = 0
):
    # GP data
    train_x = gp_model.train_inputs[0].numpy()
    train_y = gp_model.train_targets.numpy()

    # dimensions
    num_points = x_path.shape[0]
    nout = train_y.shape[1]

    x_path_tensor = torch.Tensor(x_path)
    # Make predictions
    with gpytorch.settings.fast_pred_var():
        # predictions = likelihood(gp_model(test_x))
        # predictions = gp_model(x_path_tensor) # only model (we want to find true function)
        # mean_on_path = predictions.mean.detach().numpy()
        # stddev_on_path = predictions.stddev.detach().numpy()
        # conf_lower_on_path, conf_upper_on_path = predictions.confidence_region().detach().numpy()

        # DERIVATIVE
        mean_dx = torch.autograd.functional.jacobian(
            lambda x: gp_model(x).mean.sum(dim=0), 
            x_path_tensor
        )

        # project derivative along path
        x_path_diff = x_path[1:,:]-x_path[:-1,:]
        x_path_norm = x_path_diff / np.tile(
            np.sqrt(np.sum(x_path_diff**2,axis=1)).reshape((num_points-1,1)), (1,x_path.shape[1])
        )
        mean_dx_on_path = np.array([np.inner(mean_dx[:,i,:],x_path_norm[i,:]) for i in range(num_points-1)])

        # kernel derivative
        # k = gp_model.covar_module
        # kernel_dx_left_at_train = torch.autograd.functional.jacobian(
        #     lambda x: k(x,train_x).sum(dim=0).unsqueeze(0), 
        #     x_path_tensor
        # )
        # kernel_dx_right_at_train = torch.autograd.functional.jacobian(
        #     lambda x: k(train_x,x).sum(dim=1).unsqueeze(1), 
        #     x_path_tensor
        # )
        # kernel_dx_left_at_eval = torch.autograd.functional.jacobian(
        #     lambda x: k(x,train_x).sum(dim=0).unsqueeze(0), 
        #     train_x
        # )
        # kernel_ddx_at_eval = torch.autograd.functional.jacobian(
        #     lambda x: kernel_dx_left_at_eval(x,train_x).sum(dim=0).unsqueeze(0), 
        #     train_x
        # )
        

    # draw function samples
    # sample_on_path = np.zeros((num_points,nout,num_samples))
    # for j in range(0,num_samples):
    #     sample_on_path[:,:,j] = predictions.sample()

    return GPlotData(
        x_path[1:,:],
        mean_dx_on_path,
        None,
        None,
        None,
        None,
        train_x,
        train_y
    )


def plot_gp_data(gp_data_list: List[GPlotData], 
    marker_size_lim=[5, 100],
    marker_size_reldiff_zero=1e-3,
    marker_style="x",
    plot_train_data=True,
    x_path_mode="shared"
):
    # color_list
    cmap = plt.cm.tab10  # define the colormap
    # extract all colors from the .jet map
    color_list = [cmap(i) for i in range(cmap.N)]

    # Initialize plots
    nout = gp_data_list[0].train_y.shape[1]
    fig, ax = plt.subplots(nout, 1, figsize=(8, 3*nout))
    if nout == 1:
        ax = [ax]

    
    if x_path_mode == "sequential":
        x_plot_segments_step = 1.0 / len(gp_data_list)
        x_plot_segments = np.linspace(0, 1, len(gp_data_list) + 1)
        # x_plot_all = np.linspace(0, 1, num_points * len(gp_data_list))

    for j,gp_data in enumerate(gp_data_list):
        num_points = gp_data.x_path.shape[0]
        if x_path_mode == "sequential":
            # x_plot_step = x_plot_segments_step / num_points
            # x_plot = np.arange(x_plot_segments[j], x_plot_segments[j+1], x_plot_step)
            x_plot = np.linspace(x_plot_segments[j], x_plot_segments[j+1], num_points)
        else:
            x_plot = np.linspace(0, 1, num_points)

        for i in range(0,nout):
            add_gp_plot(
                gp_data, 
                ax[i], 
                i,
                x_plot=x_plot,
                color=color_list[j],
                marker_style=marker_style,
                marker_size_lim=marker_size_lim,
                marker_size_reldiff_zero=marker_size_reldiff_zero,
                plot_train_data=plot_train_data
            )

    # general settings
    fig.set_facecolor('white')

    return fig, ax

def add_gp_plot(gp_data: GPlotData, ax, idx_out,
        x_plot=None,
        color='b',
        marker_size_lim=[5, 100],
        marker_size_reldiff_zero=1e-3,
        marker_style="x",
        plot_train_data=True
    ): 
    x_path = gp_data.x_path

    if plot_train_data:
        train_x = gp_data.train_x
        train_y_plot = gp_data.train_y[:, idx_out]

        # project on path
        train_x_on_path, train_x_dist_to_path, train_x_on_path_index = project_data_on_path(train_x, x_path)

        # square distance again for marker scaling (also quadratic)
        train_x_dist_to_path = train_x_dist_to_path**2

        # rescale to marker size limits
        train_x_dist_scale_to_zero = (train_x_dist_to_path / np.sum(train_x_on_path**2,axis=1) >= marker_size_reldiff_zero) * 1.0
        marker_size_reldiff = (train_x_dist_to_path - min(train_x_dist_to_path))/(max(train_x_dist_to_path)-min(train_x_dist_to_path))
        train_x_dist_scale = marker_size_lim[1] + (marker_size_lim[0]-marker_size_lim[1]) * (marker_size_reldiff * train_x_dist_scale_to_zero)

        num_points = x_path.shape[0]
        if x_plot is None:
            x_plot = np.linspace(0, 1, num_points)

        train_x_on_plot = x_plot[train_x_on_path_index]
        ax.scatter(train_x_on_plot, train_y_plot, 
            s=train_x_dist_scale,
            marker=marker_style,
            color=color,
            alpha=0.5
        )
    
    # Predictive mean as blue line
    mean = gp_data.mean_on_path[:, idx_out]
    ax.plot(x_plot, mean, color=color)
    
    # Shade in confidence
    # stddev = gp_data.stddev_on_path[:, idx_out]
    if (gp_data.conf_upper_on_path is not None) and \
        (gp_data.conf_lower_on_path is not None):
        upper = gp_data.conf_upper_on_path[:, idx_out]
        lower = gp_data.conf_lower_on_path[:, idx_out]
        y_plot_fill = ax.fill_between(x_plot, lower, upper, alpha=0.3, color=color)
        
    # plot samples
    if gp_data.sample_on_path is not None:
        num_samples = gp_data.sample_on_path.shape[2]
        for j in range(0,num_samples):
            sample = gp_data.sample_on_path[:, idx_out, j]
            ax.plot(x_plot, sample, color=color, linestyle=':', lw=1)

    # if gp_out_lims is not None:
    #     ax.set_ylim(gp_out_lims[i,:])

    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Observed Values (Likelihood), output {idx_out}')
    return ax

def plot_gp_model(gp_model, likelihood, x_path,
    num_samples = 0,
    marker_style = 'x',
    marker_size_lim = [1, 5],
):
    # gp_model, likelihood and x_path can be lists 
    # Initialize plots
    nout = gp_model.train_targets[0].shape[0]
    fig, ax = plt.subplots(1, nout, figsize=(8, 3))
    if nout == 1:
        ax = [ax]

    # along which dim to plot (in gp_dim_lims[dim], other values fixed to gp_dim_slice[other_dims])
    # nout = gp_model.train_targets[0].shape[0]
    num_points = x_path.shape[0]

    # Set into eval mode
    gp_model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # predictions = likelihood(gp_model(test_x))
        predictions = gp_model(torch.Tensor(x_path)) # only model (we want to find true function)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

        # draw function samples
        predictions_samples = np.zeros((num_points,nout,num_samples))
        for j in range(0,num_samples):
            predictions_samples[:,:,j] = predictions.sample()

    # loop through all outputs and plot
    for i in range(0,nout):
        # TODO: Why do I need train_inputs[0]? Some batching possibilities?        
        # plot data projected on path
        train_x = gp_model.train_inputs[0].numpy()
        train_y_plot = gp_model.train_targets[:, i].numpy()
        train_x_on_path, train_x_dist_to_path, train_x_on_path_index = project_data_on_path(train_x, x_path)

        # square distance again for marker scaling (also quadratic)
        train_x_dist_to_path = train_x_dist_to_path**2

        # rescale to marker size limits
        train_x_dist_scale = marker_size_lim[0] + (marker_size_lim[1]-marker_size_lim[0]) * (train_x_dist_to_path - min(train_x_dist_to_path))/(max(train_x_dist_to_path)-min(train_x_dist_to_path))
        test_x_plot = np.linspace(0, 1, num_points)
        train_x_on_plot = test_x_plot[train_x_on_path_index]
        ax[i].scatter(train_x_on_plot, train_y_plot, 
            s=train_x_dist_scale,
            marker=marker_style,
            color='k',
            alpha=0.5
        )
        
        # Predictive mean as blue line
        ax[i].plot(test_x_plot, mean[:, i].numpy(), 'b')
        
        # Shade in confidence
        y_plot_fill = ax[i].fill_between(test_x_plot, lower[:, i].numpy(), upper[:, i].numpy(), alpha=0.5)
        
        # plot samples
        for j in range(0,num_samples):
            ax[i].plot(test_x_plot, predictions_samples[:,i,j], 'b:', lw=1)

        # if gp_out_lims is not None:
        #     ax[i].set_ylim(gp_out_lims[i,:])

        ax[i].legend(['Observed Data', 'Mean', 'Confidence'])
        ax[i].set_title(f'Observed Values (Likelihood), output {i}')

        # gp_data = gp_data_from_model_and_path(gp_model, likelihood, x_path,
        #     num_samples = num_samples
        # )

    # general settings
    fig.set_facecolor('white')

def project_data_on_path(x_data: np.array, x_path: np.array):
    # x_path: n_plot x dim
    # x_data: n_data x dim 
    n_data = x_data.shape[0]
    n_path = x_path.shape[0]
    i_path, i_data = np.meshgrid(np.arange(0,n_path), np.arange(0,n_data))
    dist = np.sqrt(np.sum((x_data[i_data] - x_path[i_path])**2, axis=2))
    dist_min_data_index = np.argmin(dist, axis=1)
    dist_min_data = dist[np.arange(0,n_data),dist_min_data_index]
    x_data_on_path = x_path[dist_min_data_index,:]
    return x_data_on_path, dist_min_data, dist_min_data_index

def generate_grid_points(x_dim_lims, x_dim_slice, x_dim_plot, num_points=200):
    x_dim_fix = np.arange(len(x_dim_slice))
    for i in x_dim_fix:
        if i == x_dim_plot:
            x_add = np.linspace(x_dim_lims[i,0], x_dim_lims[i,1], num_points)
        else:
            x_add = x_dim_slice[i] * np.ones((num_points,))
        
        # vstack
        if i == 0:
            x_grid = x_add
        else:
            x_grid = np.vstack((x_grid, x_add))
    return x_grid.transpose()


# FROM HERE ONWARD: Functions added specifically for nonparamatric learning MPC
# Author: Mathieu Dubied (mdubied@ethz.ch)
# Date: 17/06/2024
# Function to plot GP predictions
def plot_gp_models_in_batch_quadrotor(model, likelihood, var_range, fixed_vals, var_name, n_gp_models, n_points=100, fig_size=(12,8)):
    
    # Dictionary for LaTeX formatting
    var_latex_dict = {
        'phi': '$\phi$',
        'v1': '$v_1$',
        'v2': '$v_2$'
    }
    
    # Initialiase input points at which to get GP predictions
    test_x = torch.zeros(n_points, 3)

    # Assign the range values to the appropriate variable
    if var_name == 'phi':
        test_x[:, 0] = torch.linspace(var_range[0], var_range[1], n_points)
        test_x[:, 1] = fixed_vals[0]  # v1
        test_x[:, 2] = fixed_vals[1]  # v2
    elif var_name == 'v1':
        test_x[:, 0] = fixed_vals[0]  # phi
        test_x[:, 1] = torch.linspace(var_range[0], var_range[1], n_points)
        test_x[:, 2] = fixed_vals[1]  # v2
    elif var_name == 'v2':
        test_x[:, 0] = fixed_vals[0]  # phi
        test_x[:, 1] = fixed_vals[1]  # v1
        test_x[:, 2] = torch.linspace(var_range[0], var_range[1], n_points)

    test_x = test_x.unsqueeze(0).repeat(n_gp_models, 1, 1)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    mean = observed_pred.mean.numpy()
    lower, upper = observed_pred.confidence_region()
    lower = lower.numpy()
    upper = upper.numpy()

    # Determine the number of rows and columns for subplots
    n_rows = 2
    n_cols = math.ceil(n_gp_models/2)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    if n_gp_models == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes.flat):
        ax.plot(test_x[0, :, 0].numpy() if var_name == 'phi' else (test_x[0, :, 1].numpy() if var_name == 'v1' else test_x[0, :, 2].numpy()), mean[i], 'b')
        ax.fill_between(test_x[0, :, 0].numpy() if var_name == 'phi' else (test_x[0, :, 1].numpy() if var_name == 'v1' else test_x[0, :, 2].numpy()), lower[i], upper[i], alpha=0.5)
        ax.set_title(f'GP Model {i}')
        ax.set_xlabel(f'{var_latex_dict[var_name]}')
        ax.set_ylabel('$g$')
        ax.legend(['Mean', 'CI'])
        ax.grid()

    fixed_var_names = ['phi', 'v1', 'v2']
    fixed_var_names.remove(var_name)
    fixed_vals_str = ", ".join([f"{var_latex_dict[fixed_var_names[i]]} = {fixed_vals[i]}" for i in range(2)])
    g_dep_str = f"{var_latex_dict[var_name]}"

    fig.suptitle(f"$g(${g_dep_str}$)$. The values of {fixed_vals_str} are fixed.", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    return fig, axes

def plot_gp_models_in_batch_single_input(gp_models_as_batch, n_gp_models_in_batch, train_x, train_y, ylim=None, figsize=(12,8)):
    # Determine the number of rows and columns for subplots
    n_rows = 2
    n_cols = math.ceil(n_gp_models_in_batch/2)

    # Initialize plots
    with gpytorch.settings.observation_nan_policy("fill"):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Make predictions
        with torch.no_grad():
            test_x = torch.linspace(-2, 2, 51).view(1, -1, 1).repeat(n_gp_models_in_batch, 1, 1)
            observed_pred = gp_models_as_batch(test_x[0,:,0])
            # Get mean
            mean = observed_pred.mean
            # Get lower and upper confidence bounds
            lower, upper = observed_pred.confidence_region()

    # Iterate over models
    for i, ax in enumerate(axes.flat):
        if i < n_gp_models_in_batch:
            ax.plot(train_x[i].detach().numpy(), train_y[i].detach().numpy(), 'k*')

            # Predictive mean as blue line
            ax.plot(test_x[i].squeeze().numpy(), mean[i, :].numpy(), 'b')
            # Shade in confidence
            ax.fill_between(test_x[i].squeeze().numpy(), lower[i, :].numpy(), upper[i, :].numpy(), alpha=0.5)
            # Set y-axis limits if provided
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.legend(['Observed Data', 'Mean', 'CI'])
            ax.set_title(f'GP model {i}')
            ax.grid()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    return fig, axes

def plot_gp_models_idx_in_batch_single_input(gp_models_as_batch, n_gp_models_in_batch, gp_models_idx_list, train_x, train_y,  true_function_eval, 
                                               ylim=None, figsize=(12,8), x_label='$x_2$', y_label='$g(x_2)$', legend_size=12, ax_label_size=12,tick_size=12):
    # Determine the number of rows and columns for subplots
    n_rows = 1
    n_cols = len(gp_models_idx_list)

    # Initialize plots
    with gpytorch.settings.observation_nan_policy("fill"):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Make predictions
        with torch.no_grad():
            test_x = torch.linspace(-2, 2, 51).view(1, -1, 1).repeat(n_gp_models_in_batch, 1, 1)
            observed_pred = gp_models_as_batch(test_x[0,:,0])
            # Get mean
            mean = observed_pred.mean
            # Get lower and upper confidence bounds
            lower, upper = observed_pred.confidence_region()

    # Define legend handles and labels
    legend_handles = []
    legend_labels = []

    # Iterate over models
    current_gp_idx_in_list = 0
    for i, ax in enumerate(axes.flat):
        if i < n_cols:
            current_gp_idx = gp_models_idx_list[current_gp_idx_in_list]
            observed_data_handle, = ax.plot(train_x[current_gp_idx].detach().numpy(), train_y[current_gp_idx].detach().numpy(), 'k*')
            mean_handle, = ax.plot(test_x[current_gp_idx].squeeze().numpy(), mean[current_gp_idx, :].numpy(), 'b')
            ci_handle = ax.fill_between(test_x[current_gp_idx].squeeze().numpy(), lower[current_gp_idx, :].numpy(), upper[current_gp_idx, :].numpy(), alpha=0.5)

            # Plot true function
            g_fcn_handle, = ax.plot(test_x[current_gp_idx].numpy(),true_function_eval(test_x[current_gp_idx]).numpy(),'g--')

            # Add handles to the legend only once
            if not legend_handles:
                legend_handles.extend([observed_data_handle, mean_handle, ci_handle, g_fcn_handle])
                legend_labels.extend(['Observed Data', 'Mean', 'Confidence interval', 'True function $g(x)$'])

            # Set y-axis limits if provided
            if ylim is not None:
                ax.set_ylim(ylim)

            # Axis label
            ax.set_xlabel(x_label, fontsize=ax_label_size)
            ax.set_ylabel(y_label, fontsize=ax_label_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_size)

            ax.set_title(f'GP model {current_gp_idx}')
            ax.grid()
            current_gp_idx_in_list += 1

    # Add a single legend below the figure
    fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=legend_size)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust rect to make space for the legend
    return fig, axes

def plot_single_gp_from_batch_single_input(gp_models_as_batch, n_gp_models_in_batch, train_x, train_y, true_function_eval, 
                                       figsize=(12,8), x_label='$x_2$', y_label='$g(x_2)$',
                                       legend_size=12, ax_label_size=12,tick_size=10):

    # Initialize plots
    with gpytorch.settings.observation_nan_policy("fill"):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Make predictions
        with torch.no_grad():
            test_x = torch.linspace(-2, 2, 51).view(1, -1, 1).repeat(n_gp_models_in_batch, 1, 1)
            observed_pred = gp_models_as_batch(test_x[0,:,0])
            # Get mean
            mean = observed_pred.mean
            # Get lower and upper confidence bounds
            lower, upper = observed_pred.confidence_region()

            # Plot training data as black stars
            ax.plot(train_x[0].squeeze().numpy(), train_y[0].numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(test_x[0].squeeze().numpy(), mean[0,:].numpy(), 'royalblue')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x[0].squeeze().numpy(), lower[0,:].numpy(), upper[0,:].numpy(), alpha=0.3)

            # Plot true function
            ax.plot(test_x[0].numpy(),true_function_eval(test_x[0]).numpy(),'g--')
            
            # Axis label
            ax.set_xlabel(x_label, fontsize=ax_label_size)
            ax.set_ylabel(y_label, fontsize=ax_label_size)

    # ax.set_ylim([-0.05, 0.05])
    legend = ax.legend(['Observed data', 'Mean', 'Confidence interval','True function $g(x)$'],loc='upper left', bbox_to_anchor=(1, 0.7))
    for item in legend.get_texts():
        item.set_fontsize(legend_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid()

    return fig, ax

def plot_w_bound(ax, x_plot, g_bar, observed_pred, beta, model_index):

    with torch.no_grad():
        # Get upper and lower bounds, and then bound w
        lower = g_bar - (observed_pred.mean.numpy() - beta*np.sqrt(observed_pred.variance.numpy()))
        upper = (observed_pred.mean.numpy() + beta*np.sqrt(observed_pred.variance.numpy())) - g_bar
        w = np.maximum(lower,upper) 

        # Plot the bound over x
        ax.plot(x_plot,w, label=f"GP model {model_index}")
        
    return ax

def plot_w_bound_idx_list(gp_models_as_batch,g_bar_plot, beta_plot, list_gp_idx,
                          figsize=(12,8), x_label='$x_2$', y_label='$w(x_2)$',
                          legend_size=12, ax_label_size=12, tick_size=10):
    
    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid()
    ax.set_xlabel(x_label, fontsize=ax_label_size)
    ax.set_ylabel(y_label, fontsize=ax_label_size)

    with gpytorch.settings.observation_nan_policy("fill"), torch.no_grad():
        test_x = torch.linspace(0, 2, 200)
        observed_pred = gp_models_as_batch(test_x)

    for i in range(0,len(list_gp_idx)):
        i_gp= list_gp_idx[i]
        ax = plot_w_bound(ax, test_x, g_bar_plot, observed_pred[i_gp],  beta_plot, i)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_size)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    return f, ax


## Additional functions, not used in the MT report. Allows to plot each GP model on top of each others --------------
# Might require some small adaptations
def plot_gp_model(x, pred, ax, model_index):
    with gpytorch.settings.observation_nan_policy("fill"), torch.no_grad():
        with torch.no_grad():
            # Get upper and lower confidence bounds
            lower, upper = pred.confidence_region()
            # Plot predictive means as blue line
            ax.plot(x.numpy(), pred.mean.numpy(), label=f"Mean (Model {model_index})")
            # Shade between the lower and upper confidence bounds
            ax.fill_between(x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label=f"CI (Model {model_index})")
            

    return ax

def plot_gp_models_collection_base_plot():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.grid()
    ax.set_xlabel(r'$\dot{x}$', fontsize=16)
    ax.set_ylabel(r'$g(\dot{x})$', fontsize=16)

    return ax

def plot_true_function(ax,test_x,c):
    true_function_eval = np.zeros(test_x.shape[0])
    for i in range(test_x.shape[0]):
        true_function_eval[i] = -c*test_x[i].numpy()**2

    ax.plot(test_x.numpy(),true_function_eval,'g--', label="True function")

    return ax
#----------------------------


## Additional functions, not used in the MT report. Allows to plot the index of the minimum w(x) given a GP collection ----------
# Might require some small adaptations
def plot_min_upper_max_lower(gp_models_as_batch, beta, ax):
    with gpytorch.settings.observation_nan_policy("fill"), torch.no_grad():
        with gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 2, 1000)
            observed_pred = gp_models_as_batch(test_x)

    with torch.no_grad():
        # Get upper and lower bounds
        min_upper = np.zeros((len(test_x),1))
        max_lower = np.zeros((len(test_x),1))
        idx_min_upper = np.zeros((len(test_x),1))
        idx_max_lower = np.zeros((len(test_x),1))
        for i in range(len(test_x)):
            upper_list_at_x = []
            lower_list_at_x = []
            for j in range(observed_pred.mean.shape[0]):
                upper_list_at_x.append(observed_pred.mean[j,i].numpy() + beta*np.sqrt(observed_pred.variance[j,i].numpy()))
                lower_list_at_x.append(observed_pred.mean[j,i].numpy() - beta*np.sqrt(observed_pred.variance[j,i].numpy()))
            min_upper[i] = min(upper_list_at_x)
            max_lower[i] = max(lower_list_at_x)
            idx_min_upper[i] = upper_list_at_x.index(min(upper_list_at_x))
            idx_max_lower[i] = lower_list_at_x.index(max(lower_list_at_x))
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), min_upper, label=f"Min upper")
        ax.plot(test_x.numpy(), max_lower, label=f"Max lower")
       

    return ax

def plot_min_upper_max_lower_idx(gp_models_as_batch, beta, ax):
    with gpytorch.settings.observation_nan_policy("fill"), torch.no_grad():
        with gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 2, 100)
            observed_pred = gp_models_as_batch(test_x)

    with torch.no_grad():
        # Get upper and lower bounds
        min_upper = np.zeros((len(test_x),1))
        max_lower = np.zeros((len(test_x),1))
        idx_min_upper = np.zeros((len(test_x),1))
        idx_max_lower = np.zeros((len(test_x),1))
        for i in range(len(test_x)):
            upper_list_at_x = []
            lower_list_at_x = []
            for j in range(observed_pred.mean.shape[0]):
                upper_list_at_x.append(observed_pred.mean[j,i].numpy() + beta*np.sqrt(observed_pred.variance[j,i].numpy()))
                lower_list_at_x.append(observed_pred.mean[j,i].numpy() - beta*np.sqrt(observed_pred.variance[j,i].numpy()))
            min_upper[i] = min(upper_list_at_x)
            max_lower[i] = max(lower_list_at_x)
            idx_min_upper[i] = upper_list_at_x.index(min(upper_list_at_x))
            idx_max_lower[i] = lower_list_at_x.index(max(lower_list_at_x))
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), idx_min_upper, label=f"Min upper")
        ax.plot(test_x.numpy(), idx_max_lower, label=f"Max lower")
       

    return ax

def plot_max_argument(gp_model, g_bar, beta, ax, model_index):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 2, 100)
        observed_pred = gp_model(test_x)

    with torch.no_grad():
        # Get upper and lower bounds
        lower = g_bar - ( observed_pred.mean.numpy() - beta*np.sqrt(observed_pred.variance.numpy()))
        upper = (observed_pred.mean.numpy() + beta*np.sqrt(observed_pred.variance.numpy())) - g_bar
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), lower, label=f"Lower (Model {model_index})")
        ax.plot(test_x.numpy(), upper, label=f"Upper (Model {model_index})")
        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.numpy(), lower, upper, alpha=0.2, label=f"Model {model_index}")

    return ax
#----------------------------

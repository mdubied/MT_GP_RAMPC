import numpy as np
from numpy import linalg as npla
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

@dataclass
class EllipsoidTubeData2D:
    center_data: np.ndarray = None,
    ellipsoid_data: np.ndarray = None,
    ellipsoid_colors: np.ndarray = None

class EllipsoidData2D:
    center_data: np.ndarray = None,
    delta: np.ndarray = None,
    M: np.ndarray = None
    def __init__(self, center_data,delta,M):
        self.center_data = center_data
        self.delta = delta
        self.M = M   

# Create basic state-space plot
def base_plot(lb_x1=None, ub_x1=None, lb_x2=None, ub_x2=None, size=(4,3), color='k', lw=1, x_label='$x_1$', y_label='$x_2$'):
    fig, ax = plt.subplots(1, 1, figsize=size)

    if lb_x1 is not None:
        ax.axvline(lb_x1,color=color,lw=lw)

    if ub_x1 is not None:
        ax.axvline(ub_x1,color=color,lw=lw)

    if lb_x2 is not None:
        ax.axhline(lb_x2,color=color,lw=lw)

    if ub_x2 is not None:
        ax.axhline(ub_x2,color=color,lw=lw)

    ax.set_xlabel(x_label)  # Add x-axis label
    ax.set_ylabel(y_label)  # Add y-axis label

    return fig, ax


def plot_tube(ax, z, M, delta, col='b', lw=1):
    # Generate points on unit circle
    t = np.linspace(0, 2*np.pi, 200)
    x_c = np.array([np.cos(t), np.sin(t)])

    # Scale and rotate the circle to form ellipse
    # Use Cholesky decomposition of inverse of M
    L_inv = np.linalg.cholesky(np.linalg.inv(M))
    x = np.dot(L_inv, x_c) * delta

    # Translate ellipse to position z
    x[0, :] += z[0]
    x[1, :] += z[1]

    # Plot the ellipse
    ax.plot(x[0], x[1], color=col, linewidth=lw,zorder=10)

    return ax


def add_plot_ellipse(ax,E,e0,n=50):
    # sample angle uniformly from [0,2pi] and length from [0,1]
    radius = 1.
    theta_arr = np.linspace(0,2*np.pi,n)
    w_rad_arr = [[radius, theta] for theta in theta_arr]
    w_one_arr = np.array([[w_rad[0]*np.cos(w_rad[1]), w_rad[0]*np.sin(w_rad[1])] for w_rad in w_rad_arr])
    w_ell = np.array([e0 + E @ w_one for w_one in w_one_arr])
    h = ax.plot(w_ell[:,0],w_ell[:,1],linewidth=1,zorder=10)
    return h


def add_plot_trajectory(ax,tube_data: EllipsoidTubeData2D, center_color='b', center_linestyle='-', tube_color='b', lw_center=1, lw_tube=2, z_order=None):
    n_data = tube_data.center_data.shape[0]
    evenly_spaced_interval = np.linspace(0.6, 1, n_data)
    
    if z_order is not None:
        h_plot = ax.plot(tube_data.center_data[:,0],tube_data.center_data[:,1], center_linestyle, color=center_color, linewidth=lw_center, zorder=z_order)
    else:
        h_plot = ax.plot(tube_data.center_data[:,0],tube_data.center_data[:,1], center_linestyle, color=center_color, linewidth=lw_center)
    # set color
    h_plot[0].set_color(center_color)

    if tube_data.delta is not None:
        
        for i in range(tube_data.center_data.shape[0]):
            delta_i = tube_data.delta[i]
            M = tube_data.M
            center_i = tube_data.center_data[i,:]
            ax = plot_tube(ax,center_i,M,delta_i,col=tube_color,lw=lw_tube)

    return h_plot


# Plot the solution with the tube in the p1-p2 plane, so the position of the quadrotor, with obstacle
def plot_sol_tube_obstacle(sol_data,obs_pos,
                           lb_x1=None,ub_x1=None,lb_x2=None,ub_x2=None, 
                           center_color='b', center_linestyle='-', tube_color='b', lw_center=1, lw_tube=2, 
                           size=(8,4)):
    fig, ax = base_plot(lb_x1=None,ub_x1=None,lb_x2=None,ub_x2=None,size=size)
    ax.grid()

    if obs_pos.ndim == 1:
            obs_pos = np.transpose(np.expand_dims(obs_pos,1))   # if single obstacle, make sure we get an array of size (1,3)
    for obs in obs_pos:
        x, y, radius = obs
        circle = plt.Circle((x, y), radius, edgecolor='grey', facecolor='grey', zorder=2)
        ax.add_patch(circle)

    add_plot_trajectory(ax, sol_data, center_color=center_color, center_linestyle=center_linestyle, tube_color=tube_color, lw_center=lw_center, lw_tube=lw_tube)
    
    plt.gca().set_aspect('equal')
    # plt.show()
    return ax

# Plot the state and input solutoin of the quadrotor
def plot_state_input_solution_quadrotor(X, U, N, h, figsize=(16,6)):
    """
    Plot data with the given X, U, and time_plot.

    Parameters:
    X (numpy.ndarray): Data array for X.
    U (numpy.ndarray): Data array for U.
    N (int): Number of points.
    h (float): Time step size.
    """

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    time_plot_state = np.linspace(0, N * h, N + 1)
    
    # Position
    axs[0, 0].plot(time_plot_state, X[:, 0], label='$x_1$')
    axs[0, 0].plot(time_plot_state, X[:, 1], label='$x_2$')
    axs[0, 0].set_xlabel('$t$')  
    axs[0, 0].set_ylabel('$x$')  
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    axs[0, 0].grid()

    # Velocities
    axs[1, 0].plot(time_plot_state, X[:, 3], label='$v_1$')
    axs[1, 0].plot(time_plot_state, X[:, 4], label='$v_2$')
    axs[1, 0].set_xlabel('$t$')  
    axs[1, 0].set_ylabel('$v$')  
    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    axs[1, 0].grid()

    # Angle and Angular Velocity
    axs[0, 1].plot(time_plot_state, X[:, 2], label='$\phi$')
    axs[0, 1].plot(time_plot_state, X[:, 5], label='$\dot{\phi}$')
    axs[0, 1].set_xlabel('$t$')  
    axs[0, 1].set_ylabel('$\phi, \dot{\phi}$')  
    handles, labels = axs[0, 1].get_legend_handles_labels()
    axs[0, 1].legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    axs[0, 1].grid()

    # Inputs
    time_plot_inputs = np.linspace(0, (N - 1) * h, N)
    axs[1, 1].plot(time_plot_inputs, U[:, 0], label='$u_1$')
    axs[1, 1].plot(time_plot_inputs, U[:, 1], label='$u_2$')
    axs[1, 1].set_xlabel('$t$')  
    axs[1, 1].set_ylabel('$u$')  
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles=handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    axs[1, 1].grid()
    
    return fig, axs


def plot_single_state(x, N, h, name, fig_size=(6, 4), c=None, delta=None, 
                      color_traj='royalblue', lw_traj=1, color_tube='gray', alpha_tube=0.5,
                      cl_traj=None,cl_traj_col='k',cl_traj_lw = 2):
    """
    Plot data with the given x across time and optionally draw a confidence region.

    Parameters:
    x (numpy.ndarray): Data array for the state x across time.
    N (int): Number of points.
    h (float): Time step size.
    name (str): Label for the y-axis.
    fig_size (tuple): Size of the figure.
    c (float, optional): Constant for the confidence interval.
    delta (numpy.ndarray, optional): Array for the confidence interval.
    color_traj: Color of the trajectory
    lw_traj: linewidth of the trajectory
    color_tube: color of the tube around the trajectory
    alpha_tube: transparency of the tube
    cl_traj: plot the closed-loop trajectory
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    time_plot_state = np.linspace(0, N * h, N + 1)
    
    ax.plot(time_plot_state, x, color=color_traj, linewidth=lw_traj)
    
    if c is not None and delta is not None:
        lower_bound = x - c * delta
        upper_bound = x + c * delta
        ax.fill_between(time_plot_state, lower_bound, upper_bound, color=color_tube, alpha=alpha_tube)

    if cl_traj is not None:
        ax.plot(time_plot_state, cl_traj, color=cl_traj_col, linewidth=cl_traj_lw)
    
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(name)


    return fig, ax


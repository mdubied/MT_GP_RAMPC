# %%
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, MX, DM, vertcat, sin, cos, Function, dot, fabs, mtimes,sqrt, diag
from scipy.linalg import block_diag

# %%
# TODO: change for quadrotor (Note: not used if we specify z_0=x_0 as initial condition)
def V_delta_squared(x,z,M):
    # Lyapunov function, squared
    return (x-z).T @ M @ (x-z)  

def h_obs(x,obs_pos):
    return -sqrt((x[0] - obs_pos[0])**2 + (x[1] - obs_pos[1])**2) + obs_pos[2]
# %%
def export_quadrotor_model_RMPC(constants_offline, initial_con_z=False, terminal_con=True, obstacle=True, model_name='quadrotor_model_RMPC'):
    # set up states & controls
    p1   = SX.sym('p1')
    p2  = SX.sym('p2')
    phi = SX.sym('phi')
    v1   = SX.sym('v1')
    v2  = SX.sym('v2')
    phidot = SX.sym('phidot')
    delta = SX.sym('delta')  
    x = vertcat(p1,p2,phi,v1,v2,phidot,delta)

    u1 = SX.sym('u1')
    u2 = SX.sym('u2')
    u = vertcat(u1,u2)

    n_sys_states = 6    # specific to the quadrotor system, without delta
    n_sys_inputs = 2    # specific to the quadrotor system
    
    # xdot
    p1_dot   = SX.sym('p1_dot')
    p2_dot  = SX.sym('p2_dot')
    phi_dot = SX.sym('phi_dot')
    v1_dot   = SX.sym('v1_dot')
    v2_dot  = SX.sym('v2_dot')
    phidot_dot = SX.sym('phidot_dot')
    delta_dot = SX.sym('delta_dot')
    xdot = vertcat(p1_dot,p2_dot,phi_dot,v1_dot,v2_dot,phidot_dot,delta_dot)

    # set up parameter p of the optimisation
    param1 = SX.sym('param1')
    param2 = SX.sym('param2')
    param3 = SX.sym('param3')
    param4 = SX.sym('param4')
    param5 = SX.sym('param5')
    param6 = SX.sym('param6')
    p = vertcat(param1,param2,param3,param4,param5,param6)

    # system parameters (values of b_x, b_u used in export_ocp)
    F_x = constants_offline['F_x']
    F_u = constants_offline['F_u']
    c_x = constants_offline['c_x']
    c_u = constants_offline['c_u']
    c_obs = constants_offline['c_obs']
    obs_pos = constants_offline['obs_pos']
    if obs_pos.ndim == 1:
        obs_pos = np.transpose(np.expand_dims(obs_pos,1))   # if single obstacle, make sure we get an array of size (1,3)
        c_obs = np.array(c_obs)
        c_obs = np.expand_dims(c_obs,0)
    
    F_x_casadi = SX(F_x)
    F_u_casadi = SX(F_u)

    n_const_x = len(c_x)
    n_const_u = len(c_u)
    if obstacle == True:
        n_const_obs = obs_pos.shape[0]
    else:
        n_const_obs = 0

    # constraints con_h: h(x,)_i - c_i * delta
    # states
    con_h_expr_x = []
    x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
    F_x_x = mtimes(F_x_casadi,x_sub)

    for i in range(n_const_x): 
        F_x_x_i = F_x_x[i]
        con_h_expr_x.append(F_x_x_i + c_x[i] * delta)
    con_h_expr_x = vertcat(*con_h_expr_x)

    # inputs    
    con_h_expr_u = []
    u_sub = u[0:n_sys_inputs]   # n_sys_inputs considers the 2 system's inputs, without lambdas
    F_u_u = mtimes(F_u_casadi,u_sub)

    for i in range(n_const_u):
        F_u_u_i = F_u_u[i]
        con_h_expr_u.append(F_u_u_i + c_u[i] * delta)  
    con_h_expr_u = vertcat(*con_h_expr_u) 

    # obstacle
    con_h_expr_obs = []
    if obstacle == True:
        for i in range(n_const_obs):      # obstacle constraints
            con_h_expr_obs.append(h_obs(x[0:2], obs_pos[i,:]) + c_obs[i] * delta)
        con_h_expr_obs = vertcat(*con_h_expr_obs)

    # states, inputs and obstacle together
    con_h_expr = vertcat(con_h_expr_x, con_h_expr_u, con_h_expr_obs)

    # initial constraints
    # TODO: change for M(x). V_delta needs to be rewriten using the geodesic?
    M_under = SX(constants_offline['M_under'])
    if initial_con_z == True:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:n_sys_states],x[0:n_sys_states],M_under),  
            delta,
            p1 - param1,
            p2 - param2,
            phi - param3,
            v1 - param4,
            v2 - param5,
            phidot - param6,
            delta - 0.0
        )
    else:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:n_sys_states],x[0:n_sys_states],M_under),  
            delta
        )

    # initial constraints con_h_0: states and inputs
    con_h_0 = vertcat(con_h_0,con_h_expr)

    # terminal contraints 
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        con_h_e_p1 = p1
        con_h_e_p2 = p2
        con_h_e_phi = phi
        con_h_e_v1 = v1
        con_h_e_v2 = v2
        con_h_e_phidot = phidot
        con_h_e_delta = delta

        # constraints h(x,)_i - c_i * \bar{delta}_f
        # states (no constraint on input for terminal stage, as v/u not in the terminal stage)
        con_h_e_x = []
        x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
        F_x_x = mtimes(F_x_casadi,x_sub)

        for i in range(n_const_x): 
            F_x_x_i = F_x_x[i]
            con_h_e_x.append(F_x_x_i + c_x[i] * delta_bar_f)   # use of \bar{\delta}_f
        con_h_e_x = vertcat(*con_h_e_x)

        # obstacle
        if obstacle == True:
            con_h_e_obs = []
            for i in range(n_const_obs):      # obstacle constraints
                con_h_e_obs.append(h_obs(x[0:2], obs_pos[i,:]) + c_obs[i] * delta_bar_f)
            con_h_e_obs = vertcat(*con_h_e_obs)

            #terminal constraints con_h_3: merge
            con_h_e = vertcat(con_h_e_p1, con_h_e_p2, con_h_e_phi, 
                            con_h_e_v1, con_h_e_v2, con_h_e_phidot, 
                            con_h_e_delta, con_h_e_x, con_h_e_obs)
        else:
            con_h_e = vertcat(con_h_e_p1, con_h_e_p2, con_h_e_phi, 
                        con_h_e_v1, con_h_e_v2, con_h_e_phidot, 
                        con_h_e_delta, con_h_e_x)

    # acados model
    model = AcadosModel()
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.con_h_expr = con_h_expr
    model.con_h_expr_0 = con_h_0
    if terminal_con == True:
        model.con_h_expr_e = con_h_e
    model.name = model_name


    return model


def export_ocp_quadrotor_RMPC(N, T, constants_offline, 
                              initial_con_z=False, terminal_con=True, soft_con=False, obstacle=True, 
                              x_ref = np.array([1.,2.,0.,0.,0.,0.]), delta_ref = 0., u_ref = np.array([2.4,2.4]),
                                **model_kwargs):
    
    # Generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()
    model = export_quadrotor_model_RMPC(constants_offline, initial_con_z=initial_con_z, terminal_con=terminal_con, obstacle=obstacle,  **model_kwargs)
    ocp.model = model
    ocp.dims.N = N

    # Dimensions
    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = nx + nu
    ny_0 = nx + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.nh = model.con_h_expr.shape[0]
    ocp.dims.nh_0 = model.con_h_expr_0.shape[0]
    if terminal_con == True:
        ocp.dims.nh_e = model.con_h_expr_e.shape[0]
    else:
        ocp.dims.nh_e = 0
    ocp.dims.np = model.p.shape[0] if isinstance(model.p, SX) else 0

    # Reference state
    p1_ref = x_ref[0]
    p2_ref = x_ref[1]
    phi_ref = x_ref[2]
    v1_ref = x_ref[3]
    v2_ref = x_ref[4]
    phidot_ref = x_ref[5]
    delta_ref = delta_ref
    u1_ref = u_ref[0]    # equilibrium
    u2_ref = u_ref[1]    # equilibrium

    # Cost: quadratic linear least square 
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS' 

    cost_p1 = 20
    cost_p2 = 20
    cost_phi = 10
    cost_v1 = 1
    cost_v2 = 1
    cost_phidot = 10
    cost_delta = 0.1
    cost_fac_e = 1 # TODO: change?
    Q = np.diagflat(np.array([cost_p1,cost_p2,cost_phi,cost_v1,cost_v2,cost_phidot,cost_delta]))
    R = np.diagflat(np.array([5.,5.]))

    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = cost_fac_e * Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)
    ocp.cost.Vx_0 = np.zeros((ny_0, nx))
    ocp.cost.Vx_0[:nx,:nx] = np.eye(nx)
    ocp.cost.Vx_e = np.zeros((ny_e, nx))
    ocp.cost.Vx_e[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu_0 = np.zeros((ny, nu))
    ocp.cost.Vu_0[nx:nx+nu, :] = np.eye(nu)

    ocp.cost.yref = np.array([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref,u1_ref,u2_ref])
    ocp.cost.yref_0 = np.array([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref,u1_ref,u2_ref])
    ocp.cost.yref_e = np.array([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref])


    # Cost for soft constraints formulation
    if soft_con == True:
        ocp.cost.Zu = np.eye(ocp.dims.nh)
        ocp.cost.Zl = np.eye(ocp.dims.nh)
        ocp.cost.zu = np.repeat(10000,ocp.dims.nh)
        ocp.cost.zl = np.repeat(10000, ocp.dims.nh)
        ocp.cost.Zu_0 = np.eye(ocp.dims.nh_0)
        ocp.cost.Zl_0 = np.eye(ocp.dims.nh_0)
        ocp.cost.zu_0 = np.repeat(10000,ocp.dims.nh_0)
        ocp.cost.zl_0 = np.repeat(10000, ocp.dims.nh_0)
        ocp.cost.Zu_e = np.eye(ocp.dims.nh_e)
        ocp.cost.Zl_e = np.eye(ocp.dims.nh_e)
        ocp.cost.zu_e = np.repeat(10000,ocp.dims.nh_e)
        ocp.cost.zl_e = np.repeat(10000, ocp.dims.nh_e)

        ocp.constraints.idxsh = np.arange(ocp.dims.nh)
        ocp.constraints.idxsh_0 = np.arange(ocp.dims.nh_0)
        ocp.constraints.idxsh_e = np.arange(ocp.dims.nh_e)
        print('       Type of constraints: Soft constraints')
    else:
        print('       Type of constraints: Hard constraints')
       

    # Constraints on h(x,z)
    b_x = constants_offline['b_x']
    b_u = constants_offline['b_u']
    if obstacle == True:
        obs_pos = constants_offline['obs_pos']
        obs_pos[2,1]=0.98*obs_pos[2,1]       # TODO: remove
        if obs_pos.ndim == 1:
            obs_pos = np.transpose(np.expand_dims(obs_pos,1))   # if single obstacle, make sure we get an array of size (1,3)
        
        n_const_obs = obs_pos.shape[0]
        b_obs = np.zeros((n_const_obs,))

    n_const_x = len(b_x)
    n_const_u = len(b_u)
    # 
    inf_num = 1e6
    ocp.constraints.constr_type = 'BGH'
    
    # states and inputs constraints con_h
    if obstacle == True:
        lh_expr = -inf_num*np.ones((n_const_x+n_const_u+n_const_obs,))
        uh_expr = np.concatenate((b_x, b_u,b_obs))
        print('      Obstacle constraints: Present')
    else: 
        lh_expr = -inf_num*np.ones((n_const_x+n_const_u,))
        uh_expr = np.append(b_x, b_u)
        print('      Obstacle constraints: Absent')

    ocp.constraints.lh = lh_expr
    ocp.constraints.uh = uh_expr
    
    # initial constraints con_h_0
    lh_0_V_delta = 0.0 
    lh_0_delta = 0.0 
    if initial_con_z == True:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta, 0., 0., 0., 0., 0., 0.,0.]) # Lyapunov function, delta, and the 6 states, + delta again
        uh_0 = np.array([inf_num, inf_num, 0., 0., 0., 0., 0., 0.,0.])
        print(' Initial stage constraints: Present, with explicit constraints on z')
    else:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta])
        uh_0 = np.array([inf_num, inf_num])
        print(' Initial stage constraints: Present, without explicit constraints on z')

    ocp.constraints.lh_0 = np.append(lh_0,lh_expr)
    ocp.constraints.uh_0 = np.append(uh_0,uh_expr)

    # terminal constraints: z1, z2, delta, con_h_e_expr_x
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        if obstacle == True:
            lh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, 0, -inf_num*np.ones((n_const_x+n_const_obs,))))  
            uh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, delta_bar_f,b_x,b_obs )) 
        else:
            lh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, 0, -inf_num*np.ones((n_const_x,))))  
            uh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, delta_bar_f,b_x )) 
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
        print('Terminal stage constraints: Present')
    else: 
        print('Terminal stage constraints: Absent')

    # Initialisation of the parameter
    ocp.parameter_values = np.repeat(0.0,nx-1)    # delta not included

    # solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.tf = T
        
    return ocp


#  %%
def export_quadrotor_model_RAMPC(constants_offline, num_lambdas, initial_con_z=False, terminal_con=True, obstacle=True, model_name='quadrotor_model_RAMPC'):
    # set up states & controls
    p1   = SX.sym('p1')
    p2  = SX.sym('p2')
    phi = SX.sym('phi')
    v1   = SX.sym('v1')
    v2  = SX.sym('v2')
    phidot = SX.sym('phidot')
    delta = SX.sym('delta')  
    x = vertcat(p1,p2,phi,v1,v2,phidot,delta)

    u1 = SX.sym('u1')
    u2 = SX.sym('u2')
    lambdas = [SX.sym(f'lmbda{i+1}') for i in range(num_lambdas)]
    u = vertcat(u1, u2, *lambdas)

    n_sys_states = 6    # specific to the quadrotor system, without delta
    n_sys_inputs = 2    # specific to the quadrotor system
    
    # xdot
    p1_dot   = SX.sym('p1_dot')
    p2_dot  = SX.sym('p2_dot')
    phi_dot = SX.sym('phi_dot')
    v1_dot   = SX.sym('v1_dot')
    v2_dot  = SX.sym('v2_dot')
    phidot_dot = SX.sym('phidot_dot')
    delta_dot = SX.sym('delta_dot')
    xdot = vertcat(p1_dot,p2_dot,phi_dot,v1_dot,v2_dot,phidot_dot,delta_dot)

    # set up parameter p of the optimization
    param1 = SX.sym('param1')
    param2 = SX.sym('param2')
    param3 = SX.sym('param3')
    param4 = SX.sym('param4')
    param5 = SX.sym('param5')
    param6 = SX.sym('param6')
    p = vertcat(param1,param2,param3,param4,param5,param6)

    # system parameters (values of b_x, b_u used in export_ocp)
    F_x = constants_offline['F_x']
    F_u = constants_offline['F_u']
    c_x = constants_offline['c_x']
    c_u = constants_offline['c_u']
    c_obs = constants_offline['c_obs']
    obs_pos = constants_offline['obs_pos']
    if obs_pos.ndim == 1:
        obs_pos = np.transpose(np.expand_dims(obs_pos,1))   # if single obstacle, make sure we get an array of size (1,3)
        c_obs = np.array(c_obs)
        c_obs = np.expand_dims(c_obs,0)


    F_x_casadi = SX(F_x)
    F_u_casadi = SX(F_u)

    n_const_x = len(c_x)
    n_const_u = len(c_u)
    if obstacle == True:
        n_const_obs = obs_pos.shape[0]
    else:
        n_const_obs = 0

    # constraints h(x,)_i - c_i * delta
    # states
    con_h_expr_x = []
    x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
    F_x_x = mtimes(F_x_casadi,x_sub)

    for i in range(n_const_x): 
        F_x_x_i = F_x_x[i]
        con_h_expr_x.append(F_x_x_i + c_x[i] * delta)
    con_h_expr_x = vertcat(*con_h_expr_x)

    # inputs    
    con_h_expr_u = []
    u_sub = u[0:n_sys_inputs]   # n_sys_inputs considers the 2 system's inputs, without lambdas
    F_u_u = mtimes(F_u_casadi,u_sub)

    for i in range(n_const_u):
        F_u_u_i = F_u_u[i]
        con_h_expr_u.append(F_u_u_i + c_u[i] * delta)

    for i in range(num_lambdas):
        con_h_expr_u.append(u[n_sys_inputs+i])
        
    con_h_expr_u = vertcat(*con_h_expr_u)

    # obstacle
    con_h_expr_obs = []
    if obstacle == True:
        for i in range(n_const_obs):      # obstacle constraints
            con_h_expr_obs.append(h_obs(x[0:2], obs_pos[i,:]) + c_obs[i] * delta)
        con_h_expr_obs = vertcat(*con_h_expr_obs)
    # states, inputs and obstacle together
    con_h_expr = vertcat(con_h_expr_x, con_h_expr_u, con_h_expr_obs)

    # initial constraints
    # TODO: change for M(x). V_delta needs to be rewriten using the geodesic?
    M_under = SX(constants_offline['M_under'])
    if initial_con_z == True:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:n_sys_states],x[0:n_sys_states],M_under),  
            delta,
            p1 - param1,
            p2 - param2,
            phi - param3,
            v1 - param4,
            v2 - param5,
            phidot - param6
        )
    else:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:n_sys_states],x[0:n_sys_states],M_under),  
            delta
        )

    # initial constraints: states and inputs
    con_h_0 = vertcat(con_h_0, con_h_expr)

    # terminal contraints 
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        con_h_e_p1 = p1
        con_h_e_p2 = p2
        con_h_e_phi = phi
        con_h_e_v1 = v1
        con_h_e_v2 = v2
        con_h_e_phidot = phidot
        con_h_e_delta = delta

        # constraints h(x,)_i - c_i * \bar{delta}_f
        # states (no constraint on input for terminal stage, as v/u not in the terminal stage)
        con_h_e_x = []
        x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
        F_x_x = mtimes(F_x_casadi,x_sub)

        for i in range(n_const_x): 
            F_x_x_i = F_x_x[i]
            con_h_e_x.append(F_x_x_i + c_x[i] * delta_bar_f)   # use of \bar{\delta}_f
        con_h_e_x = vertcat(*con_h_e_x)

        # obstacle
        con_h_e_obs = []
        for i in range(n_const_obs):      # obstacle constraints
            con_h_e_obs.append(h_obs(x[0:2], obs_pos[i,:]) + c_obs[i] * delta_bar_f)
        con_h_e_obs = vertcat(*con_h_e_obs)

        # terminal constraints con_h_3: merge
        con_h_e = vertcat(con_h_e_p1, con_h_e_p2, con_h_e_phi, 
                          con_h_e_v1, con_h_e_v2, con_h_e_phidot, 
                          con_h_e_delta, con_h_e_x, con_h_e_obs)

    # acados model
    model = AcadosModel()
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.con_h_expr = con_h_expr
    model.con_h_expr_0 = con_h_0
    if terminal_con == True:
        model.con_h_expr_e = con_h_e
    model.name = model_name

    return model


def export_ocp_quadrotor_RAMPC(N, T, constants_offline, num_lambdas, initial_con_z=False, terminal_con=True, soft_con=False,  obstacle=True, **model_kwargs):

    # Generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()
    model = export_quadrotor_model_RAMPC(constants_offline, num_lambdas, initial_con_z=initial_con_z, terminal_con=terminal_con, obstacle=obstacle, **model_kwargs)
    ocp.model = model
    ocp.dims.N = N

    # Dimensions
    nx = model.x.shape[0]
    nu = model.u.shape[0]
    ny = nx + nu
    ny_0 = nx + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.nh = model.con_h_expr.shape[0]
    ocp.dims.nh_0 = model.con_h_expr_0.shape[0]
    if terminal_con == True:
        ocp.dims.nh_e = model.con_h_expr_e.shape[0]
    else:
        ocp.dims.nh_e = 0
    ocp.dims.np = model.p.shape[0] if isinstance(model.p, SX) else 0

    # Reference state
    p1_ref = 1.0
    p2_ref = 2.0
    phi_ref = 0.
    v1_ref = 0.
    v2_ref = 0.
    phidot_ref = 0.
    delta_ref = 0.
    u1_ref = 2.4    # equilibrium
    u2_ref = 2.4    # equilibrium

    # Cost: quadratic linear least square 
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS' 

    cost_p1 = 20
    cost_p2 = 20
    cost_phi = 10
    cost_v1 = 5
    cost_v2 = 5
    cost_phidot = 10
    cost_delta = 0.1
    cost_fac_e = 10 # TODO: change?
    Q = np.diagflat(np.array([cost_p1,cost_p2,cost_phi,cost_v1,cost_v2,cost_phidot,cost_delta]))
    R = np.diagflat(np.concatenate(([5.,5.], np.repeat(0.1,num_lambdas))))

    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = cost_fac_e * Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)
    ocp.cost.Vx_0 = np.zeros((ny_0, nx))
    ocp.cost.Vx_0[:nx,:nx] = np.eye(nx)
    ocp.cost.Vx_e = np.zeros((ny_e, nx))
    ocp.cost.Vx_e[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu_0 = np.zeros((ny, nu))
    ocp.cost.Vu_0[nx:nx+nu, :] = np.eye(nu)

    ocp.cost.yref = np.concatenate(([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref,u1_ref,u2_ref], np.repeat(0.0,num_lambdas)))  
    ocp.cost.yref_0 = np.concatenate(([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref,u1_ref,u2_ref], np.repeat(0.0,num_lambdas)))  
    ocp.cost.yref_e = np.array([p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref,delta_ref])


    # Cost for soft constraints formulation
    if soft_con == True:
        ocp.cost.Zu = np.eye(ocp.dims.nh)
        ocp.cost.Zl = np.eye(ocp.dims.nh)
        ocp.cost.zu = np.repeat(10000,ocp.dims.nh)
        ocp.cost.zl = np.repeat(10000, ocp.dims.nh)
        ocp.cost.Zu_0 = np.eye(ocp.dims.nh_0)
        ocp.cost.Zl_0 = np.eye(ocp.dims.nh_0)
        ocp.cost.zu_0 = np.repeat(10000,ocp.dims.nh_0)
        ocp.cost.zl_0 = np.repeat(10000, ocp.dims.nh_0)
        ocp.cost.Zu_e = np.eye(ocp.dims.nh_e)
        ocp.cost.Zl_e = np.eye(ocp.dims.nh_e)
        ocp.cost.zu_e = np.repeat(10000,ocp.dims.nh_e)
        ocp.cost.zl_e = np.repeat(10000, ocp.dims.nh_e)

        ocp.constraints.idxsh = np.arange(ocp.dims.nh)
        ocp.constraints.idxsh_0 = np.arange(ocp.dims.nh_0)
        ocp.constraints.idxsh_e = np.arange(ocp.dims.nh_e)
        print('       Type of constraints: Soft constraints')
    else:
        print('       Type of constraints: Hard constraints')
       


    # constraints on h(x,z): Order is Fx<b_x,Fu<bu,<b_lambda_lb<lambda<b_lambda_ub, h_obstacle(x)<b_obs
    b_x = constants_offline['b_x']
    b_u = constants_offline['b_u']
    if obstacle == True:
        obs_pos = constants_offline['obs_pos']
        if obs_pos.ndim == 1:
            obs_pos = np.transpose(np.expand_dims(obs_pos,1))   # if single obstacle, make sure we get an array of size (1,3)
        
        n_const_obs = obs_pos.shape[0]
        b_obs = np.zeros((n_const_obs,))

    n_const_x = len(b_x)
    n_const_u = len(b_u)
    n_const_lambda = num_lambdas
    if obstacle == True:
        n_const_obs = obs_pos.shape[0]
    else:
        n_const_obs = 0
    b_lambda_lb = np.repeat(1.0,num_lambdas)      
    b_lambda_ub = np.repeat(1.0,num_lambdas)
    b_obs = np.zeros((n_const_obs,))
    inf_num = 1e6
    ocp.constraints.constr_type = 'BGH'
    
    # states and inputs TODO: add a if statement for b_obs
     # states and inputs constraints con_h
    if obstacle == True:
        lh_expr = -inf_num*np.ones((n_const_x+n_const_u+n_const_lambda+n_const_obs,))  # constraint on lambda after constraint on system input u
        lh_expr[n_const_x+n_const_u:n_const_x+n_const_u+n_const_lambda] = b_lambda_lb  # constraint on lambda after constraint on system input u
        uh_expr = np.concatenate((b_x, b_u, b_lambda_ub, b_obs))
        print('      Obstacle constraints: Present')
    else: 
        lh_expr = -inf_num*np.ones((n_const_x+n_const_u+n_const_lambda,))  # constraint on lambda after constraint on system input u
        lh_expr[n_const_x+n_const_u:n_const_x+n_const_u+n_const_lambda] = b_lambda_lb  # constraint on lambda after constraint on system input u
        uh_expr = np.concatenate((b_x, b_u, b_lambda_ub))
        print('      Obstacle constraints: Absent')

    ocp.constraints.lh = lh_expr
    ocp.constraints.uh = uh_expr

    # initial constraints
    lh_0_V_delta = 0.0 
    lh_0_delta = 0.0 

    if initial_con_z == True:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta, 0., 0., 0., 0., 0., 0.]) # Lyapunov function, delta, and the 6 states
        uh_0 = np.array([inf_num, inf_num, 0., 0., 0., 0., 0., 0.])
        print(' Initial stage constraints: Present, with explicit constraints on z')
    else:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta])
        uh_0 = np.array([inf_num, inf_num])
        print(' Initial stage constraints: Present, without explicit constraints on z')

    ocp.constraints.lh_0 = np.append(lh_0,lh_expr)
    ocp.constraints.uh_0 = np.append(uh_0,uh_expr)

    # terminal constraints: z1, z2, delta, con_h_e_expr_x
    # TODO: change for no obstacle
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        if obstacle == True:
            lh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, 0, -inf_num*np.ones((n_const_x+n_const_obs,))))  
            uh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, delta_bar_f,b_x,b_obs )) 
        else:
            lh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, 0, -inf_num*np.ones((n_const_x,))))  
            uh_e = np.hstack((p1_ref,p2_ref,phi_ref,v1_ref,v2_ref,phidot_ref, delta_bar_f,b_x )) 
        
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
        print('Terminal stage constraints: Present')
    else:
        print('Terminal stage constraints: Absent')

   
    # ocp.constraints.x0 = x0 # no fix initial state 
    ocp.parameter_values = np.repeat(0.0,nx-1)    # delta not included

    # solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.tf = T

    return ocp


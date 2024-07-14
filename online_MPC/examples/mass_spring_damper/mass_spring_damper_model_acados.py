# %%
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, MX, DM, vertcat, sin, cos, Function, dot, fabs, mtimes, diag, blockcat
from scipy.linalg import block_diag

# %%
def V_delta_squared(x,z,M):
    # Lyapunov function, squared
    return (x-z).T @ M @ (x-z)  
# %%
# %%
def export_MSDS_model_RMPC(constants_offline, initial_con_z=False, terminal_con=True, model_name='massspringdamper_model_RMPC'):
    # set up states & controls
    z1   = SX.sym('z1')
    z2  = SX.sym('z2')
    delta = SX.sym('delta')
    
    x = vertcat(z1, z2, delta)
    u = SX.sym('u')

    n_sys_states = 2    # specific to the mass spring damper system, without delta
    n_sys_inputs = 1    # specific to the mass spring damper system
    
    # xdot
    z1_dot   = SX.sym('z1_dot')
    z2_dot  = SX.sym('z2_dot')
    delta_dot = SX.sym('delta_dot')
    xdot = vertcat(z1_dot, z2_dot, delta_dot)

    # set up parameter p of the optimisation
    p1 = SX.sym('p1')
    p2 = SX.sym('p2')
    p = vertcat(p1,p2)

    # system parameters (values of b_x, b_u used in export_ocp)
    F_x = constants_offline['F_x']
    F_u = constants_offline['F_u']
    c_x = constants_offline['c_x']
    c_u = constants_offline['c_u']
    
    F_x_casadi = SX(F_x)
    F_u_casadi = SX(F_u)

    n_const_x = len(c_x)
    n_const_u = len(c_u)

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

    # constraints con_h: states and inputs together
    con_h_expr = vertcat(con_h_expr_x, con_h_expr_u)

    # initial constraints
    M = SX(constants_offline['M'])
    if initial_con_z == True:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:2],x[0:2],M),  
            delta,
            z1-p1,
            z2-p2
        )
    else:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:2],x[0:2],M),  
            delta
        )

    # initial constraints con_h_0: states and inputs
    con_h_0 = vertcat(con_h_0,con_h_expr)

    # terminal contraints 
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        con_h_e_z1 = z1
        con_h_e_z2 = z2
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

        # terminal constraints con_h_3: merge
        con_h_e = vertcat(con_h_e_z1, con_h_e_z2, con_h_e_delta, con_h_e_x)
    
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


def export_ocp_MSDS_RMPC(N, T, constants_offline, initial_con_z=False, terminal_con=True, soft_con=False, 
                         z_ref=np.array([3.,0.]), **model_kwargs):
    
    # Generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()
    model = export_MSDS_model_RMPC(constants_offline, initial_con_z=initial_con_z, terminal_con=terminal_con,  **model_kwargs)
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
    z1_ref = z_ref[0]
    z2_ref = z_ref[1]

    # Cost: quadratic linear least square 
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS' 

    cost_x1 = 10
    cost_x2 = 6.
    cost_delta = 0.01 
    cost_fac_e = 1. 
    Q = np.diagflat(np.array([cost_x1, cost_x2, cost_delta]))
    R = np.array(1.)

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

    ocp.cost.yref = np.array([z1_ref,z2_ref,0.,0.])
    ocp.cost.yref_0 = np.array([z1_ref,z2_ref,0.,0.])
    ocp.cost.yref_e = np.array([z1_ref,z2_ref,0.])


    # cost for soft constraints formulation
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
    n_const_x = len(b_x)
    n_const_u = len(b_u)
    inf_num = 1e6
    ocp.constraints.constr_type = 'BGH'
    
    # states and inputs constraints con_h
    lh_expr = -inf_num*np.ones((n_const_x+n_const_u,))
    uh_expr = np.append(b_x, b_u)
    ocp.constraints.lh = lh_expr
    ocp.constraints.uh = uh_expr
    
    # initial constraints con_h_0
    lh_0_V_delta = 0.0 
    lh_0_delta = 0.0 
    if initial_con_z == True:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta, 0., 0.])
        uh_0 = np.array([inf_num, inf_num, 0., 0.])
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
        lh_e = np.hstack((z1_ref,z2_ref, 0.,-inf_num*np.ones((n_const_x,)))) 
        uh_e = np.hstack((z1_ref,z2_ref, delta_bar_f,b_x))
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
        print('Terminal stage constraints: Present')
    else:
        print('Terminal stage constraints: Absent')

    # Initialisation of the parameter
    ocp.parameter_values = np.array([0.,0.])

    # solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.tf = T
        
    return ocp


#  %%

def export_MSDS_model_RAMPC(constants_offline, num_lambdas, initial_con_z=False, terminal_con=True, model_name='massspringdamper_model_RAMPC'):
    # set up states & controls
    z1 = SX.sym('z1')
    z2 = SX.sym('z2')
    delta = SX.sym('delta')
    x = vertcat(z1, z2, delta)

    v = SX.sym('v')
    lambdas = [SX.sym(f'lmbda{i+1}') for i in range(num_lambdas)]
    u = vertcat(v, *lambdas)

    n_sys_states = 2    # specific to the mass spring damper system, without delta
    n_sys_inputs = 1    # specific to the mass spring damper system, without lambda
    
    # xdot
    z1_dot = SX.sym('z1_dot')
    z2_dot = SX.sym('z2_dot')
    delta_dot = SX.sym('delta_dot')
    xdot = vertcat(z1_dot, z2_dot, delta_dot)

    # set up parameter p of the optimization
    p1 = SX.sym('p1')
    p2 = SX.sym('p2')
    p = vertcat(p1, p2)

    # system parameters (values of b_x, b_u used in export_ocp)
    F_x = constants_offline['F_x']
    F_u = constants_offline['F_u']
    c_x = constants_offline['c_x']
    c_u = constants_offline['c_u']
    
    F_x_casadi = SX(F_x)
    F_u_casadi = SX(F_u)

    n_const_x = len(c_x)
    n_const_u = len(c_u)

    # constraints con_h: h(x,)_i - c_i * delta
    # states
    con_h_expr_x = []
    x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
    F_x_x = mtimes(F_x_casadi,x_sub)

    for i in range(n_const_x): 
        F_x_x_i = F_x_x[i]
        con_h_expr_x.append(F_x_x_i + c_x[i] * delta)
    con_h_expr_x = vertcat(*con_h_expr_x)

    # inputs: system inputs   
    con_h_expr_u = []
    u_sub = u[0:n_sys_inputs]   # n_sys_inputs considers the 2 system's inputs, without lambdas
    F_u_u = mtimes(F_u_casadi,u_sub)

    for i in range(n_const_u):
        F_u_u_i = F_u_u[i]
        con_h_expr_u.append(F_u_u_i + c_u[i] * delta)

    # inputs: lambda inputs/decision variables
    for i in range(num_lambdas):
        con_h_expr_u.append(lambdas[i])
        
    con_h_expr_u = vertcat(*con_h_expr_u) 


    # constraints con_h: states and inputs together
    con_h_expr = vertcat(con_h_expr_x, con_h_expr_u)

    # initial constraints
    M = SX(constants_offline['M'])
    if initial_con_z == True:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:2],x[0:2],M),  
            delta,
            z1-p1,
            z2-p2
        )
    else:
        con_h_0 = vertcat(
            delta**2- V_delta_squared(p[0:2],x[0:2],M),  
            delta
        )

    # initial constraints con_h_0: states and inputs
    con_h_0 = vertcat(con_h_0,con_h_expr)

    # terminal contraints 
    if terminal_con == True:
        delta_bar_f = constants_offline['delta_bar_0']
        con_h_e_z1 = z1
        con_h_e_z2 = z2
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

        # terminal constraints con_h_3: merge
        con_h_e = vertcat(con_h_e_z1, con_h_e_z2, con_h_e_delta, con_h_e_x)

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


def export_ocp_MSDS_RAMPC(N, T, constants_offline, num_lambdas, initial_con_z=False, terminal_con=True, soft_con=False, 
                          z_ref=np.array([3.,0.]), **model_kwargs):

    # Generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()
    model = export_MSDS_model_RAMPC(constants_offline, num_lambdas, initial_con_z=initial_con_z, terminal_con=terminal_con, **model_kwargs)
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
    ocp.dims.nh_e = model.con_h_expr_e.shape[0]
    ocp.dims.np = model.p.shape[0] if isinstance(model.p, SX) else 0

    # Reference state
    z1_ref = z_ref[0]
    z2_ref = z_ref[1]


    # Cost: quadratic linear least square
    cost_x1 = 10.
    cost_x2 = 6.
    cost_delta = 0.01
    cost_fac_e = 1. 
    Q = np.diagflat(np.array([cost_x1, cost_x2, cost_delta]))
    R = np.diagflat(np.concatenate(([1.], np.repeat(1e-1,num_lambdas)))) # small cost on lambda 
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
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

    ocp.cost.yref = np.concatenate(([z1_ref, z2_ref, 0.0, 0.0], np.repeat(0.0,num_lambdas)))
    ocp.cost.yref_0 = np.concatenate(([z1_ref, z2_ref, 0.0, 0.0], np.repeat(0.0,num_lambdas)))
    ocp.cost.yref_e = np.array([z1_ref, z2_ref, 0.0])  

    # cost for soft constraints formulation
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
    n_const_x = len(b_x)
    n_const_u = len(b_u)
    inf_num = 1e6
    ocp.constraints.constr_type = 'BGH'

   # states and inputs constraints con_h
    lh_expr = -inf_num*np.ones((n_const_x+n_const_u,))
    lh_expr = np.append(lh_expr, np.zeros(num_lambdas))     # add lower bound 0 on lambdas
    uh_expr = np.append(b_x, b_u)
    uh_expr = np.append(uh_expr, np.ones(num_lambdas))      # add upper bound 1 on lambdas
    ocp.constraints.lh = lh_expr
    ocp.constraints.uh = uh_expr
    
    # initial constraints con_h_0
    lh_0_V_delta = 0.0 
    lh_0_delta = 0.0 
    if initial_con_z == True:
        lh_0 = np.array([lh_0_V_delta, lh_0_delta, 0., 0.])
        uh_0 = np.array([inf_num, inf_num, 0., 0.])
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
        lh_e = np.hstack((z1_ref,z2_ref, 0.,-inf_num*np.ones((n_const_x,))))
        uh_e = np.hstack((z1_ref,z2_ref, delta_bar_f,b_x))
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
        print('Terminal stage constraints: Present')
    else:
        print('Terminal stage constraints: Absent')

    # Initialisation of the parameter
    ocp.parameter_values = np.array([0.,0.])
    
    # solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = T

    return ocp


# %%
def export_MSDS_model_no_tube(constants_offline, terminal_con=True, model_name='massspringdamper_model_no_tube'):
    # set up states & controls
    z1   = SX.sym('z1')
    z2  = SX.sym('z2')
    
    x = vertcat(z1, z2)
    u = SX.sym('u')

    n_sys_states = 2    # specific to the mass spring damper system, without delta
    n_sys_inputs = 1    # specific to the mass spring damper system
    
    # xdot
    z1_dot   = SX.sym('z1_dot')
    z2_dot  = SX.sym('z2_dot')

    xdot = vertcat(z1_dot, z2_dot)

    # set up parameter p of the optimisation
    p1 = SX.sym('p1')
    p2 = SX.sym('p2')
    p = vertcat(p1,p2)

    # system parameters (values of b_x, b_u used in export_ocp)
    F_x = constants_offline['F_x']
    F_u = constants_offline['F_u']
    c_x = constants_offline['c_x']
    c_u = constants_offline['c_u']
    
    F_x_casadi = SX(F_x)
    F_u_casadi = SX(F_u)

    n_const_x = len(c_x)
    n_const_u = len(c_u)

    # constraints con_h: h(x,)_i - c_i * delta
    # states
    con_h_expr_x = []
    x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
    F_x_x = mtimes(F_x_casadi,x_sub)

    for i in range(n_const_x): 
        F_x_x_i = F_x_x[i]
        con_h_expr_x.append(F_x_x_i)
    con_h_expr_x = vertcat(*con_h_expr_x)

    # inputs    
    con_h_expr_u = []
    u_sub = u[0:n_sys_inputs]   # n_sys_inputs considers the 2 system's inputs, without lambdas
    F_u_u = mtimes(F_u_casadi,u_sub)

    for i in range(n_const_u):
        F_u_u_i = F_u_u[i]
        con_h_expr_u.append(F_u_u_i)
        
    con_h_expr_u = vertcat(*con_h_expr_u) 

    # constraints con_h: states and inputs together
    con_h_expr = vertcat(con_h_expr_x, con_h_expr_u)

    # initial constraints
    con_h_0 = vertcat(
        z1-p1,
        z2-p2
    )


    # initial constraints con_h_0: states and inputs
    con_h_0 = vertcat(con_h_0,con_h_expr)

    # terminal contraints 
    if terminal_con == True:
        con_h_e_z1 = z1
        con_h_e_z2 = z2

        # constraints h(x,)_i - c_i * \bar{delta}_f
        # states (no constraint on input for terminal stage, as v/u not in the terminal stage)
        con_h_e_x = []
        x_sub = x[0:n_sys_states]   # n_sys_states considers the 6 system's states, without delta
        F_x_x = mtimes(F_x_casadi,x_sub)

        for i in range(n_const_x): 
            F_x_x_i = F_x_x[i]
            con_h_e_x.append(F_x_x_i)   # use of \bar{\delta}_f
        con_h_e_x = vertcat(*con_h_e_x)

        # terminal constraints con_h_3: merge
        con_h_e = vertcat(con_h_e_z1, con_h_e_z2, con_h_e_x)
    
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


def export_ocp_MSDS_nominal(N, T, constants_offline, terminal_con=True, soft_con=False, z_ref=np.array([3.,0.]), **model_kwargs):
    
    # Generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()
    model = export_MSDS_model_no_tube(constants_offline,terminal_con=terminal_con, **model_kwargs)
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
    z1_ref = z_ref[0]
    z2_ref = z_ref[1]

    # Cost: quadratic linear least square 
    ocp.cost.cost_type_0 = 'LINEAR_LS'
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS' 

    cost_x1 = 10
    cost_x2 = 6.
    cost_fac_e = 1. 
    Q = np.diagflat(np.array([cost_x1, cost_x2]))
    R = np.array(1.)

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

    ocp.cost.yref = np.array([z1_ref,z2_ref,0.])
    ocp.cost.yref_0 = np.array([z1_ref,z2_ref,0.])
    ocp.cost.yref_e = np.array([z1_ref,z2_ref])


    # cost for soft constraints formulation
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
    n_const_x = len(b_x)
    n_const_u = len(b_u)
    inf_num = 1e6
    ocp.constraints.constr_type = 'BGH'
    
    # states and inputs constraints con_h
    lh_expr = -inf_num*np.ones((n_const_x+n_const_u,))
    uh_expr = np.append(b_x, b_u)
    ocp.constraints.lh = lh_expr
    ocp.constraints.uh = uh_expr
    
    # initial constraints con_h_0
    lh_0 = np.array([0., 0.])
    uh_0 = np.array([0., 0.])
    print(' Initial stage constraints: Present, with explicit constraints on z')
    
    ocp.constraints.lh_0 = np.append(lh_0,lh_expr)
    ocp.constraints.uh_0 = np.append(uh_0,uh_expr)

    # terminal constraints: z1, z2, delta, con_h_e_expr_x
    if terminal_con == True:
        lh_e = np.hstack((z1_ref,z2_ref, -inf_num*np.ones((n_const_x,)))) 
        uh_e = np.hstack((z1_ref,z2_ref, b_x))
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
        print('Terminal stage constraints: Present')
    else:
        print('Terminal stage constraints: Absent')

    # Initialisation of the parameter
    ocp.parameter_values = np.array([0.,0.])

    # solver options
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.tf = T
        
    return ocp

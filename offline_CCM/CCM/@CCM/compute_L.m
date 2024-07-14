%% compute_L
% -------------------------------------------------------------------------
% File: compute_L.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 29th May 2024
% License: MIT
% Description: given the (true) function g (g_fcn) and the bounds
% specificied in params (specifically F_x and b_x), compute the Lipschitz
% constant L for which ||g(x1)-g(x2)|| <= L ||x1-x2||_M. In this context,
% Lg is the Lipschitz constant for which ||g(x1)-g(x2)|| <= Lg ||x1-x2||.
% Lg is computed as Lg = max_x{||\frac{\partial g}{\partial x}||}
%
% -------------------------------------------------------------------------
function obj = compute_L(obj,params,g_fcn,M_under_chol)
    fprintf('            L: Global search starting ...')
    nx = size(params.x_ref,1);
    F_x = params.F_x;
    b_x = params.b_x;
    x_sym = sym('x', [nx, 1]);
    rng default % For reproducibility 

    % Compute the Jacobian of g_fcn
    dg_dx_fcn = jacobian(g_fcn(x_sym), x_sym);
    
    % Convert symbolic Jacobian to MATLAB function handle
    dg_dx_fcn_handle = matlabFunction(dg_dx_fcn, 'Vars', {x_sym});

    % Objective function for maximization: L2 norm of the Jacobian
    obj_max = @(x) -norm(dg_dx_fcn_handle(x), 2);
    
    
    gs = GlobalSearch('Display','off');

    % Maximize the norm of dg_dx_fcn
    problem = createOptimProblem('fmincon', 'x0', zeros(6, 1),...
        'objective', obj_max, 'Aineq', F_x, 'bineq', b_x);
    [~, fval_max, exitflag_max, ~] = run(gs, problem);

    if exitflag_max < 1
        error('The computation of L could not be correctly computed (Global search failed).')
    end

    % Since we minimized -norm, we need to negate the result to get the maximum norm
    Lg = -fval_max;
    
    % Compute L based on Lg and M_under^{1/2}
    
    L = Lg*1/min(svd(M_under_chol));
    
    
    fprintf(' done!\n')
    fprintf('              - Lg = %g\n', Lg);
    fprintf('              - L = %g\n', L);
    
    obj.L = L;
end
%% computeBoundsG
% -------------------------------------------------------------------------
% File: computeBoungsG.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 24/06/2024
% License: MIT
% Description: given the (true) function g (g_fcn) and the bounds
% specificied in params (specifically F_x and b_x), compute the maximal and
% minimal value of each component of \frac{\partial g}{\partial x}
%
% Note: only tested for scalar function g
%
% -------------------------------------------------------------------------
function bounds = computeBoundsG(params,g_fcn)
    fprintf('Computing bounds on dg/dx ...')
    nx = size(params.x_ref,1);
    F_x = params.F_x;
    b_x = params.b_x;
    x_sym = sym('x', [nx, 1]);
    rng default % For reproducibility 
    
    % Compute the Jacobian of g_fcn
    dg_dx_fcn = jacobian(g_fcn(x_sym), x_sym);
    
    % Convert symbolic Jacobian to MATLAB function handle
    dg_dx_fcn_handle = matlabFunction(dg_dx_fcn, 'Vars', {x_sym});
    
    % Number of elements in the Jacobian
    nElements = numel(dg_dx_fcn);
    
    % Initialize the vector to store the upper and lower bounds
    upperBounds = zeros(1, nElements);
    lowerBounds = zeros(1, nElements);
   

    % Loop over Jacobian elements
    for i = 1:nElements
        % Objective functions for maximization and minimization
        obj_max = @(x) -evaluateJacobianElement(dg_dx_fcn_handle, x, i);
        obj_min = @(x) evaluateJacobianElement(dg_dx_fcn_handle, x, i);
           
        gs = GlobalSearch('Display','off');

        % Maximize dg_dx_fcn(i)
        problem = createOptimProblem('fmincon','x0',zeros(nx,1),...
            'objective',obj_max, 'Aineq', F_x, 'bineq', b_x);
        [x_max,fval_max,exitflag_max,~] = run(gs,problem);
        
        % Minimize dg_dx_fcn(i)
        problem = createOptimProblem('fmincon','x0',zeros(nx,1),...
            'objective',obj_min, 'Aineq', F_x, 'bineq', b_x);
        [x_min,fval_min,exitflag_min,~] = run(gs,problem);
        
        if exitflag_min<1 || exitflag_max<1
            error('One of the bound on dg/dx could not be correctly computed (Global search failed).')
        end
        
        % Store upper and lower bound
        upperBounds(i) = -fval_max; % we minimised the negative of dg/dx
        lowerBounds(i) = fval_min;
        
    end
    
    % Display results
    fprintf(' done!\n')
    fprintf('Upper bounds: [')
    fprintf('%g ', upperBounds);
    fprintf(']\n');
    fprintf('Lower bounds: [')
    fprintf('%g ', lowerBounds);
    fprintf(']\n');
    
    % Create a single return object
    bounds(:,:,1) = upperBounds;
    bounds(:,:,2) = lowerBounds;
end

function val = evaluateJacobianElement(dg_dx_fcn_handle, x, idx)
    jacobianValues = dg_dx_fcn_handle(x);
    val = jacobianValues(idx);
end



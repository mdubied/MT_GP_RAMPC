%% compute_E_M
% -------------------------------------------------------------------------
% File: compute_E_M.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 29th May 2024
% License: MIT
% Description: compute the constant max{M^{1/2}Ed} for state dependent M
% and E, and bounded scalar d
% Note: adapted from the function compute_L_D by Alexander Erdin
%
% -------------------------------------------------------------------------
function obj = compute_E_M(obj,x_sym,d_sym,M_chol_fcn,Ed_fcn)
    % Determine states used / needed in Ed = E*d
    vars_used = findDependencies(Ed_fcn(x_sym,d_sym),x_sym,d_sym);
    vars_used{1}(obj.W_idxs) = false;
    
    % Compute n_vars
    n_vars = cellfun(@sum, vars_used);
    n_vars = [sum(obj.W_idxs); n_vars(:)];
    
    % Compute number of grid points according to grid pattern
    n_grid = getNgrid(n_vars,obj.grid_pattern([1,2,5]),obj.n_tot.E_M,'n_grid',[0,0,2]);
    
    % Get samples from state polyhedron (needed for E_M)
    x_param = getSamples(obj.params.F_x,obj.params.b_x,n_grid(1),obj.W_idxs);
    x_eval = getSamples(obj.params.F_x,obj.params.b_x,n_grid(2),vars_used{1});
    n_param = size(x_param,2);
    
    % Get all vertex combinations of d (enter E_M linearly)
    d_eval = getCombinations(obj.params.disturbance_v.*vars_used{2});

    % Create additional variables to prevent communication overhead for parfor
    W_idxs = obj.W_idxs;
    M_chol_fcn = @(x) M_chol_fcn(x);
    Ed_fcn = @(x,d) Ed_fcn(x,d);

    % Create dataqueue for parfor progbar
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) progbar('increment',100/n_param));
    
    % Initialize progress bar
    progbar(0,'prefix','          E_M: ')
    
    % Loop over parameter samples
    E_M = -inf;
    parfor k = 1:n_param
        % Get current state
        x = x_param(:,k);
        
        % Compute Cholesky decomposition of M(x)
        M_chol = full(M_chol_fcn(x));
    
        % Loop over state samples
        for i = 1:size(x_eval,2)
            % Permute x_eval with x
            x_i = x_eval(:,i);
            x_i(W_idxs) = x(W_idxs);
    
            % Loop over d
            for j = 1:size(d_eval,2)
                d_j = d_eval(:,j);
    
                % Compute E*d
                Ed = full(Ed_fcn(x_i,d_j));
               
                % Find maximum continuity constant
                E_M = max(E_M, norm(M_chol*Ed,2));
                
            end
        end
        % Update progress bar
        send(q,k);
    end
    obj.E_M = E_M;
end
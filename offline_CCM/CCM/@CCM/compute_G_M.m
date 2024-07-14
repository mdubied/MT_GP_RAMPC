%% compute_G_M
% -------------------------------------------------------------------------
% File: compute_G_M.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 29th May 2024
% License: MIT
% Description: compute the constant max{M^{1/2}G} for state dependent M and
% G
% Note: adapted from the function compute_L_G by Alexander Erdin
%
% -------------------------------------------------------------------------
function obj = compute_G_M(obj,x_sym,u_sym,M_chol_fcn,G_fcn)
    % Determine states used / needed in Gs
    vars_used = findDependencies(G_fcn(x_sym,u_sym),x_sym,u_sym);
    vars_used{1}(obj.W_idxs) = false;
    
    % Compute n_vars
    n_vars = cellfun(@sum, vars_used);
    n_vars = [sum(obj.W_idxs); n_vars(:)];
    
    % Compute number of grid points according to grid pattern
    n_grid = getNgrid(n_vars,obj.grid_pattern(1:3),obj.n_tot.G_M,'n_grid',[0,0,0]);
    
    % Get samples from state and input polyhedron (needed for G_M)
    x_param = getSamples(obj.params.F_x,obj.params.b_x,n_grid(1),obj.W_idxs);
    x_eval = getSamples(obj.params.F_x,obj.params.b_x,n_grid(2),vars_used{1});
    u_eval = getSamples(obj.params.F_u,obj.params.b_u,n_grid(3),vars_used{2});
    n_param = size(x_param,2);

    % Create additional variables to prevent communication overhead for parfor
    W_idxs = obj.W_idxs;
    M_chol_fcn = @(x) M_chol_fcn(x);
    G_fcn = @(x,u) G_fcn(x,u);
    np = obj.np;
    
    % Create dataqueue for parfor progbar
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) progbar('increment',100/n_param));
    
    % Initialize progress bar
    progbar(0,'prefix','          G_M: ')
    
    % Loop over parameter samples
    G_M = -inf(obj.np,n_param);
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
            
            % Loop over input samples
            for j = 1:size(u_eval,2)
                u_j = u_eval(:,j);
                    
                % Compute Gs
                G = full(G_fcn(x_i,u_j));

                % Find maximum continuity constant
                G_M = max(G_M, norm(M_chol*G,2));

            end
        end
        % Update progress bar
        send(q,k);
    end
    obj.G_M = max(G_M,[],2);
end
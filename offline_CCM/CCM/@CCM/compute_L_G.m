%% compute_L_G
% -------------------------------------------------------------------------
% File: compute_L_G.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 08 June 2024
% License: MIT
% Description: compute the constant L_G = max_z{\norm{\partial
% (M_cholG(g-\mu))/\partial{x} (M_chol(z))^{-1}}
% For now tailored to constant G, and mu=0
% TODO: make sure the function can deal with other cases as well, e.g., non
% constant G.
%
% Note: adapted from the initial function compute_L_G by Alexander Erdin
%
% -------------------------------------------------------------------------
function obj = compute_L_G(obj,x_sym,u_sym,M_chol_fcn,G_fcn)
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
    
    % Get all vertex combinations of g_bounds (enter dynamics linearly)
    g_bounds = getCombinations(squeeze(reshape(obj.sys.g_bound,[],1,2)));
    n_g = size(g_bounds,2);
    nx = obj.nx;
    
    % Create dataqueue for parfor progbar
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) progbar('increment',100/n_param));
    
    % Initialize progress bar
    progbar(0,'prefix','          L_G: ')
    
    
    L_G = -inf(obj.np,n_param*n_param);
    
    % Loop over parameter samples x
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
                
                
                % Loop over parameters samples: x'
                for k2 = 1:n_param
                    % Get current state
                    x2 = x_param(:,k2);

                    % Compute Cholesky decomposition of M(x)
                    M_chol2 = full(M_chol_fcn(x2));

                    % Loop over state samples
                    for i2 = 1:size(x_eval,2)
                        % Permute x_eval with x
                        x_i2 = x_eval(:,i2);
                        x_i2(W_idxs) = x(W_idxs);

                        % Loop over the combinations of dg/dx bounds
                        for m = 1:n_g
                            dg_dx_m = reshape(g_bounds(:,m),[],nx);
                            current_value = norm(M_chol*G*dg_dx_m*inv(M_chol2));
                            
                            % Find maximum value
                            L_G = max(L_G, current_value);
                        end % m
                                      
                    end % i2
                end % k2

            end % j
        end % i
        % Update progress bar
        send(q,k);
    end
    obj.L_G = max(L_G,[],2);
end
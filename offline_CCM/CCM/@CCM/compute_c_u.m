function obj = compute_c_u(obj,M_chol_fcn,hs_u_fcn)
    % Compute n_vars
    n_vars = sum(obj.W_idxs);
    
    % Compute number of grid points according to grid pattern
    n_grid = getNgrid(n_vars,obj.grid_pattern(1),obj.n_tot.c_u);
    
    % Get samples from state polyhedron for parametrization
    x_param = getSamples(obj.params.F_x,obj.params.b_x,n_grid(1),obj.W_idxs);
    n_param = size(x_param,2);

    % Create additional variables to prevent communication overhead for parfor
    M_chol_fcn = @(x) M_chol_fcn(x);
    hs_u_fcn = @(x) hs_u_fcn(x);

    % Create dataqueue for parfor progbar
    q = parallel.pool.DataQueue;
    afterEach(q, @(~) progbar('increment',100/n_param));
    
    % Initialize progress bar
    progbar(0,'prefix','       Inputs: ')
    
    % Loop over parameter samples
    c_u = -inf(size(obj.params.F_u,1),1);
    parfor k = 1:n_param
        % Get current state
        x = x_param(:,k);
        
        % Compute Cholesky decomposition of M(x)
        M_chol = full(M_chol_fcn(x));
        
        % Compute dh_u/du * K(z)
        hs_u = full(hs_u_fcn(x));
        
        % Find maximum continuity constant
        c_u = max(c_u, vecnorm(hs_u/M_chol,2,2));
        
        % Update progress bar
        send(q,k);
    end
    obj.c_u = c_u;
end
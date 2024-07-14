function [obj, success] = computeConstants(obj,recompute)
    arguments
        obj (1,1) CCM
        recompute (1,1) struct = struct('rccm',false,'c_x',false,'c_u',false,'c_obs',false,'E_M',false,'L_G',false,'delta_over',false,'M_under',false)
    end
    % Create symbolic variables for state and input
    import casadi.*
    x_sym = SX.sym('x',obj.nx);
    u_sym = SX.sym('u',obj.nu);
    d_sym = SX.sym('d',obj.nw);
    theta_sym = SX.sym('theta',obj.np);
    
    % Get W and Y
    W = obj.W_fcn(x_sym);
    Y = obj.Y_fcn(x_sym);
    
    % Compute Cholesky decomposition A = R'*R
    W_chol = chol(W);
    M_chol = chol(inv(W));
    
    % Get E and G
    E = obj.sys.E_fcn(x_sym);
    G = obj.sys.G_fcn(x_sym,u_sym);
    
    % Define functions for M^1/2 and W^1/2
    M_chol_fcn = Function('M_chol',{x_sym},{M_chol});
    W_chol_fcn = Function('W_chol',{x_sym},{W_chol});
    
    % Define functions for computation of c_j
    hs_x_fcn = Function('hs_x',{x_sym},{SX(obj.params.F_x)});
    hs_u_fcn = Function('hs_u',{x_sym},{obj.params.F_u*Y/W});
    if isa(obj.params.h_obs,"function_handle")
        hs_obs_fcn = Function('hs_ob',{x_sym},{jacobian(obj.params.h_obs(x_sym),x_sym)});
    end
    
    % Define functions for computation of E_M and L_G
    E_fcn = Function('E',{x_sym,d_sym},{E*d_sym});
    G_fcn = Function('G',{x_sym,u_sym},{G});
%     Es_fcn = Function('Es',{x_sym,d_sym},{jacobian(M_chol*E*d_sym,x_sym)}); 
    Gs_fcn = Function('Gs',{x_sym,u_sym,theta_sym},{jacobian(M_chol*G*theta_sym,x_sym) + jacobian(M_chol*G*theta_sym,u_sym)*Y/W});
    Gs_fcn_custom = Function('Gs_custom',{x_sym,u_sym},{jacobian(M_chol*G,x_sym)});  
    %% Load Constants
    vars = ["c_x","c_u","E_M","delta_over","M_under"];
    if obj.params.param_uncertainty; vars = [vars,"L_G"]; end
    
    % Try to load constants
    [obj, constants_loaded] = obj.loadVars('constants',vars,["W_coef","Y_coef"]);
    
    %% Compute Tightenings
    fprintf('Computing c_j\n');

    % Try to load c_x
    if constants_loaded && ~recompute.c_x
        fprintf('       States: loaded\n');
    else % compute c_x
        startParpool();
        obj = obj.compute_c_x(M_chol_fcn,hs_x_fcn);
    end

    % Try to load c_u
    if constants_loaded && ~recompute.c_u
        fprintf('       Inputs: loaded\n');
    else % compute c_u
        startParpool();
        obj = obj.compute_c_u(M_chol_fcn,hs_u_fcn);
    end

    % Try to load c_obs
    if isa(obj.params.h_obs,"function_handle")
        [obj, c_obs_loaded] = obj.loadVars('constants',"c_obs","W_coef");
        if c_obs_loaded && ~recompute.c_obs
            fprintf('    Obstacles: loaded\n');
        else % compute c_obs
            startParpool();
            obj = obj.compute_c_obs(x_sym,M_chol_fcn,hs_obs_fcn);
        end
    end
    
    %% Compute M_under for Tube Representation
    if obj.monomial_degree > 0
        fprintf('Computing M_under\n');
        % Try to load M_under
        if constants_loaded && ~recompute.M_under
            fprintf('      M_under: loaded\n');
        else % compute M_under
            startParpool();
            obj = obj.computeMunder();
        end
    end
    
    
    %% Compute Continuity Constants
    fprintf('Computing L, G_M, E_M \n')
    
    paramSystem = PQRparams();
    g_fcn_handle = @(x) obj.sys.g_fcn(x);
    
    % Lipschitz constant L
%     obj = obj.compute_L(paramSystem,g_fcn_handle,chol(obj.M_under));
    
    % Try to load L_G
    if constants_loaded && ~recompute.L_G
        fprintf('          L_G: loaded\n');
    else % compute L_G
        startParpool();
        obj = obj.compute_L_G(x_sym,u_sym,M_chol_fcn,G_fcn);
    end
    
    % Try to load G_M
    if constants_loaded && ~recompute.G_M
        fprintf('          G_M: loaded\n');
    else % compute G_M
        startParpool();
        obj = obj.compute_G_M(x_sym,u_sym,M_chol_fcn,G_fcn);
    end
   
    % Try to load E_M
    if constants_loaded && ~recompute.E_M
        fprintf('          E_M: loaded\n');
    else % compute E_M
        startParpool();
        obj = obj.compute_E_M(x_sym,d_sym,M_chol_fcn,E_fcn);
    end
    
    
    %% Delta_over for Rigid Tube Formulation
    % Try to load delta_over
%     if constants_loaded && ~recompute.delta_over
%         fprintf('   delta_over: loaded\n');
%     else % compute delta_over
%         startParpool();
%         obj = obj.compute_delta_over(x_sym,u_sym,theta_sym,d_sym,W_chol_fcn);
%     end
    
 
    %% Check Constants
    % Check terminal constraint
    z_s0 = obj.params.x_ref;
    v_s0 = obj.params.u_ref(z_s0,zeros(obj.np,1));
    
    % Compute maximal \delta that satisfies constraints h_j(z_s0,v_s0) + c_j delta_bar <= 0
    obj.delta_bar_x = min((obj.params.b_x - obj.params.F_x*z_s0)./obj.c_x);
    obj.delta_bar_u = min((obj.params.b_u - obj.params.F_u*v_s0)./obj.c_u);
    delta_bar_0 = min(obj.delta_bar_x,obj.delta_bar_u);
    
    [d_delta, delta_inf] = obj.checkConstants(z_s0,v_s0,delta_bar_0,M_chol_fcn);
    obj.delta_bar_0 = delta_bar_0;
    obj.delta_inf = delta_inf;


    fprintf('Effective contraction rate for uncertain nonlinear system (Prop. 3): %.4f\n', obj.rho-obj.L_G);
    
    if d_delta <= 0
        obj.terminal_invariance = true;
        fprintf('Terminal invariance condition is satisfied f_delta = %5.4f <= 0\n',d_delta);
    else
        obj.terminal_invariance = false;
        fprintf(2,'Warning: Terminal invariance condition is not satisfied f_delta = %5.4f > 0\n',d_delta);
        if obj.terminal_constraint
            success = false;
            return
        end
    end

    %% Save Constants
    success = true;
    vars = ["c_x","c_u","E_M","G_M","delta_over","M_under"];
    props = ["W_coef","Y_coef","params.F_x","params.b_x","params.F_u","params.b_u","params.theta_v","params.w_max"];
    if ~isempty(obj.c_obs)
        vars = [vars,"c_obs"];
    end
%     if obj.params.param_uncertainty
%         vars = [vars,"L_G"];
%     end
    obj.saveVars('constants',vars,props);
end
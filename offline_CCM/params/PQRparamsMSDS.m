function params = PQRparamsMSDS()
% PQRPARAMS initializes a struct containing the controller parameters for 
% a 1D mass spring damper system.
    
    params = Parameters();
    params.label = 'Mass spring damper system';
    params.version = 1;
    
    % Horizon and discrete step time
    params.N = 40;
    params.dt = 0.1;
    
    % Reference state and input
    k = 1; 
    m = 1;
    params.x_ref = [3;0];
    params.u_ref = @(x,theta) k/m*params.x_ref(1);  % equilibrium
    params.x0 = [0;0];
    
    % Constraints and vertices
    params.F_u = [eye(1); -eye(1)];
    params.b_u = [10,10]';
    params.F_x = blkdiag([eye(2); -eye(2)]);
    params.b_x = [10, 2, 10, 2]';
    params.theta_v = zeros(1,2);    % parameter between 0 and 0, not used
    params.theta_true = 0;          % not needed for our formulaiton
    params.w_max = 0.05;
    params.nw = 1;
    
%     % Cost % TODO: update with final value (nee)
%     params.Q_cost = eye(6);
%     params.R_cost = eye(2);
%     params.P_cost = 30*params.Q_cost;
%     params.regularizer = 1E-6;
%     
%     % Integrator
%     params.integrator = "rk4";
end
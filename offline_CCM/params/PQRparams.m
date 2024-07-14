function params = PQRparams()
% PQRPARAMS initializes a struct containing the controller parameters for 
% a planar quadrotor.
    
    params = Parameters();
    params.label = 'Planar Quadrotor';
    params.version = 2;
    
    % Horizon and discrete step time
    params.N = 10;
    params.dt = 0.075;
    
    % Reference state and input
    params.x_ref = zeros(6,1);
    params.u_ref = @(x,theta) zeros(2,1);
    params.x0 = [0;0;0;0;0;0];
    
    % Constraints and vertices
    params.F_u = [eye(2); -eye(2)];
    params.b_u = [3.5,3.5,1,1]';
    params.F_x = blkdiag([eye(2); -eye(2)], [eye(4); -eye(4)]);
    params.b_x = [[20, 20, 20, 20],         [pi/8, 1, 1, pi/3, pi/8, 1, 1, pi/3]]';
    params.theta_v = zeros(1,2);
    params.theta_true = 0;
    params.w_max = 0.02;
    params.nw = 1;
    
    % Obstacle constraints
    obs = [1.0   1.25    0.16;
           1.5   1.25    0.16;
           1.2   1.75    0.16];
    params.obs_pos = obs;
    params.h_obs = @(x) -sqrt((x(1) - obs(:,1)).^2 + (x(2) - obs(:,2)).^2) + obs(:,3);

    
    % Cost
    params.Q_cost = eye(6);
    params.R_cost = eye(2);
    params.P_cost = 30*params.Q_cost;
    params.regularizer = 1E-6;
    
    % Integrator
    params.integrator = "rk4";
end
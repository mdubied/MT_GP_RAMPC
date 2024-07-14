% -------------------------------------------------------------------------
% File: PlanarQuadrotor.m
% Author: Alexander Erdin (aerdin@ethz.ch)
% Date: 21th April 2024
% Date: 19th April 2024
% Modified: Mathieu Dubied (mdubied@ethz.ch)
% Date: 27th May 2024
% License: MIT
% Reference:
%
% -------------------------------------------------------------------------
classdef PlanarQuadrotor < NonlinearSystem
    
    properties (GetAccess=public,SetAccess=protected)
        nx = 6              % Number of states
        nu = 2              % Number of inputs
        np = 1              % Number of parametric uncertainties
        nw = 1              % Number of disturbance states
        ng = 1              % Number of gp outputs
        dt                  % Sampling interval

        g_bound             % Lower and upper bound on g(x)
        
        g = 9.81            % [m/s^2] gravity
        l = 0.25            % [m] half-width of quad rotor
        m = 0.4811          % [kg] true mass of the quad rotor
        m_nom = 0.486       % [kg] nominal mass of the quad rotor
        J = 0.00383         % [kgm^2] moment of inertia
        
%         plot = PlanarQuadrotorPlots() % Class for plotting
    end
    
    methods
        % state : p_x, p_z, phi, p_x_dot, p_z_dot, phi_dot
        function obj = initialize(obj,dt,options)
            arguments
                obj (1,1) PlanarQuadrotor
                dt (1,1) {mustBeNumeric, mustBePositive}
                options.integrator (1,1) string {mustBeMember(options.integrator,["multi","single","rk4"])} = 'rk4'
            end
            
            % Set sampling interval
            obj.dt = dt;
            
            % Define integrator
            obj.integrator = options.integrator;
            
            % Compute upper and lower bound on dg/dx
            planarQuadrotorParam = PQRparams();
            obj.g_bound = zeros(obj.ng,obj.nx,2);
            g_fcn_handle = @(x) obj.g_fcn(x);
            obj.g_bound = computeBoundsG(planarQuadrotorParam, g_fcn_handle);  
        end
        
        function dt = f_fcn(obj,x) % State dynamics
            dt = [x(4,:).*1 - x(5,:).*x(3,:);                          % px
                  x(4,:).*obj.sinx(x(3,:)) + x(5,:).*obj.cosx(x(3,:)); % pz
                  x(6,:);                                              % phi
                  x(6,:).*x(5,:) - obj.g*obj.sinx(x(3,:));             % vx
                 -x(6,:).*x(4,:) - obj.g*obj.cosx(x(3,:));             % vz
                  zeros(1,size(x,2))];
        end
        
        function dt = B_fcn(obj,x) % Input dynamics
            dt = [zeros(4,2);
                  1/obj.m_nom,  1/obj.m_nom;                           % vz
                  obj.l/obj.J, -obj.l/obj.J];                          % phi_dot
        end
        
        function dt = G_fcn(obj,x,u) % Parameter dynamics of the continuouse time system
            dt = [1;
                  0;
                  0;
                  0;
                  0;
                  0];
        end
        
        function dt = E_fcn(obj,x) % Disturbance dynamics
            dt = [0;
                  0;
                  0;
                  obj.cosx(x(3));
                 -obj.sinx(x(3));
                  0];
        end
        
        function dt = g_fcn(obj,x)  % True function g (used to compute bounds on dg/dx)
            dt = x(4) * (cos(x(3)) - 1) - x(5) * (sin(x(3)) - x(3));
%             dt = x(4,:)*cos(x(3,:))-x(5,:)*sin(x(3,:)) - (-x(5,:) + cos(x(3,:))-sin(x(3,:)));
        end
    end
end
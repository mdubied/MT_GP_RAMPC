% -------------------------------------------------------------------------
% File: MassSpringDamperSystem.m
% Author: Mathieu Dubied (mdubied@ethz.ch)
% Date: 24/06/2024
% License: MIT
%
% -------------------------------------------------------------------------
classdef MassSpringDamperSystem < NonlinearSystem
    
    properties (GetAccess=public,SetAccess=protected)
        nx = 2              % Number of states
        nu = 1              % Number of inputs
        np = 1              % Number of parametric uncertainties
        nw = 1              % Number of disturbance states
        ng = 1              % Number of gp outputs
        dt                  % Sampling interval

        g_bound             % Lower and upper bound on g(x)
        
        g = 9.81            % [m/s^2] gravity
        k = 1               % [N/m] elastic constant
        m = 1               % [kg] true mass of the system
        c = 0.1            % [kg/m] damping factor
  
    end
    
    methods
        % state : p_x, p_z, phi, p_x_dot, p_z_dot, phi_dot
        function obj = initialize(obj,dt,options)
            arguments
                obj (1,1) MassSpringDamperSystem
                dt (1,1) {mustBeNumeric, mustBePositive}
                options.integrator (1,1) string {mustBeMember(options.integrator,["multi","single","rk4"])} = 'rk4'
            end
            
            % Set sampling interval
            obj.dt = dt;
            
            % Define integrator
            obj.integrator = options.integrator;
            
            % Compute upper and lower bound on dg/dx
            MSDSParam = PQRparamsMSDS();
            obj.g_bound = zeros(obj.ng,obj.nx,2);
            g_fcn_handle = @(x) obj.g_fcn(x);
            obj.g_bound = computeBoundsG(MSDSParam, g_fcn_handle);  
        end
        
        function dt = f_fcn(obj,x) % State dynamics
            dt = [x(2,:);                   % x
                  -obj.k/obj.m * x(1,:)];   % \dot{x}        
        end
        
        function dt = B_fcn(obj,x) % Input dynamics
            dt = [0;
                  1];
        end
        
        function dt = G_fcn(obj,x,u) % Parameter dynamics of the continuouse time system
            dt = [0;
                  1];
        end
        
        function dt = E_fcn(obj,x) % Disturbance dynamics
            dt = [0;
                  1];
        end
        
        function dt = g_fcn(obj,x)  % True function g (used to compute bounds on dg/dx)
            dt = -obj.c*x(2).^2;
        end
    end
end
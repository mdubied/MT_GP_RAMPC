% -------------------------------------------------------------------------
% File: NonlinearSystem.m
% Author: Alexander Erdin (aerdin@ethz.ch)
% Date: 21th April 2024
% Date: 19th April 2024
% Modified: Mathieu Dubied (mdubied@ethz.ch)
% Date: 26th May 2024
% License: MIT
% Reference:
%
% -------------------------------------------------------------------------
classdef (Abstract) NonlinearSystem < handle
    
    properties (Abstract,GetAccess=public,SetAccess=protected)
        nx (1,1) {mustBeInteger,mustBeNonnegative}                         % Number of states
        nu (1,1) {mustBeInteger,mustBeNonnegative}                         % Number of inputs
        np (1,1) {mustBeInteger,mustBeNonnegative}                         % Number of parametric uncertainties
        nw (1,1) {mustBeInteger,mustBeNonnegative}                         % Number of disturbance states
        dt (1,1) {mustBeNumeric}                                           % Sampling interval
        
%         plot (1,1)                                                         % Class for plotting
    end

    properties (GetAccess=public,SetAccess=protected)
        A (1,1) {mustBeA(A,["casadi.Function","double"])} = NaN            % State matrix of the discrete time linearized dynamics
        B (1,1) {mustBeA(B,["casadi.Function","double"])} = NaN            % Input matrix of the discrete time linearized dynamics
        A_theta (1,1) {mustBeA(A_theta,["casadi.Function","double"])} = NaN% State matrix of the discrete time linearized theta dynamics
        B_theta (1,1) {mustBeA(B_theta,["casadi.Function","double"])} = NaN% Input matrix of the discrete time linearized theta dynamics
        H (1,1) {mustBeA(H,["casadi.Function","double"])} = NaN            % Hessian
        A_diff (1,1) {mustBeA(A_diff,["function_handle","double"])} = NaN  % Differnetial state dynamics
        B_diff (1,1) {mustBeA(B_diff,["function_handle","double"])} = NaN  % Differential input dynamics
        BwXw (1,1) {mustBeA(BwXw,["function_handle","double"])} = NaN      % Differential disturbance dynamics
        integrator (1,1) string {mustBeMember(integrator,["multi","single","rk4"])} = "multi"
        param_uncertainty (1,1) logical                                    % Boolean for parametric uncertainty
    end
    
    properties (Access=protected,Hidden)
        approximate (1,1) logical = false                                  % Boolean used for approximate function evaluation
    end
    
    methods
        function obj = NonlinearSystem(varargin,options)
            arguments (Repeating)
                varargin
            end
            arguments
                options.approximate (1,1) logical = true
                options.param_uncertainty (1,1) logical = true
            end
            
            % Initialize properties
            obj = obj.initialize(varargin{:});
            
            % Define parametric uncertainty (true/false)
            obj.param_uncertainty = options.param_uncertainty;
            
            % Define integrator (depending on parametric uncertainty)
            if obj.param_uncertainty
                if ~strcmp(obj.integrator, 'single')
                    fprintf(2,'Warning: Change integrator to euler single for nonlinear systems with parametric uncertainty!\n')
                end
            end

            % Check for casadi
            if ~exist('casadi.MX', 'class')
                id = 'NonlinearSystem:UndefinedClass';
                error(id,'CASADI not found. Add CASADI to the MATLAB search path.')
            end
            import casadi.*

            % Define symbolic variables
            x_sym = SX.sym('x',obj.nx);
            u_sym = SX.sym('u',obj.nu);
            theta_sym = SX.sym('theta',obj.np);
            sym_vars = {x_sym,u_sym};
            
            % Compute discrete time linearized dynamics
            obj.A = Function('A', sym_vars, {jacobian(obj.ddyn(x_sym,u_sym), x_sym)});
            obj.B = Function('B', sym_vars, {jacobian(obj.ddyn(x_sym,u_sym), u_sym)});
            
            % Compute discrete time linearized theta dynamics
            obj.A_theta = Function('A_theta', sym_vars, {jacobian(obj.ddyn_theta(x_sym,u_sym), x_sym)});
            obj.B_theta = Function('B_theta', sym_vars, {jacobian(obj.ddyn_theta(x_sym,u_sym), u_sym)});

            % Compute Hessian
            sym_vars = [x_sym;u_sym];
            obj.H = Function('H',{x_sym,u_sym,theta_sym}, {jacobian(jacobian(obj.ddyn(x_sym,u_sym,'theta',theta_sym), sym_vars), sym_vars)}); 
            
            % Compute differential dynamics
            obj.computeDifferentialDynamics(options.approximate);
        end
        
        function dt = fw(obj,x,u,theta,d) % True continuous-time dynamics of system
            dt = obj.f_fcn(x) + obj.B_fcn(x)*u;

            % Add parametric uncertainty
            if ~isequal(theta,false) && obj.param_uncertainty
                dt = dt + obj.G_fcn(x,u)*theta;
            end
            
            % Add noise
            if ~isequal(d,false)
                dt = dt + obj.E_fcn(x)*d;
            end
        end
        
        function x_p = ddyn(obj,x,u,options) % Discretized dynamics
            arguments
                obj (1,1) NonlinearSystem
                x (:,:) {mustBeVector}
                u (:,:) {mustBeVector}
                options.dynamics (1,1) = false
                options.theta (:,:) {mustBeVector} = false
                options.d (:,:) {mustBeVector} = false
                options.dt (1,1) {mustBePositive} = obj.dt
            end
            % Set dynamics
            h = options.dt;
            if isa(options.dynamics,'function_handle')
                dynamics = options.dynamics;
            else
                dynamics = @(x,u) obj.fw(x,u,options.theta,options.d);
            end
            % Choose discretization
            switch obj.integrator
                case 'single'
                    x_p = x + h*dynamics(x,u);
                case 'multi'
                    step = 10;
                    for i = 1:step
                        x = x + h/step*dynamics(x,u);
                    end
                    x_p = x;
                case 'rk4'
                    k_1 = dynamics(x,u);
                    k_2 = dynamics(x+0.5*h*k_1,u);
                    k_3 = dynamics(x+0.5*h*k_2,u);
                    k_4 = dynamics(x+h*k_3,u);
                    x_p = x + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;
                otherwise
                    error('unrecognised integrator');
            end
        end
        
        function x_p = ddyn_theta(obj,x,u) % Discretized parametric uncertainty dynamics
            % Set dynamics of parametric uncertainty
            if obj.param_uncertainty
                dynamics = @(x,u) obj.G_fcn(x,u);
            else
                dynamics = @(x,u) zeros(obj.np,1); % set to zero
            end
            x_p = obj.ddyn(x,u,'dynamics',dynamics);
        end

        function computeDifferentialDynamics(obj,approximate) 
            x_sym = sym('x',[obj.nx,1]);
            u_sym = sym('u',[obj.nu,1]);
            d_sym = sym('d',[obj.nw,1]);
            theta_sym = sym('theta',[obj.np,1]);
            sym_vars = {x_sym,u_sym,theta_sym,d_sym};
            
            % Compute differential dynamics
            obj.approximate = approximate;
            A_diff = jacobian(obj.fw(sym_vars{:}),x_sym); %#ok
            B_diff = jacobian(obj.fw(sym_vars{:}),u_sym); %#ok
            BwXw = obj.E_fcn(x_sym)*d_sym; %#ok
            if obj.param_uncertainty
                BwXw = BwXw + obj.G_fcn(x_sym,u_sym)*theta_sym; %#ok
            end
            obj.approximate = false;
            
            % Convert to variable-precision arithmetic representation
            A_diff = vpa(A_diff); %#ok
            B_diff = vpa(B_diff); %#ok
            BwXw = vpa(BwXw);     %#ok
            
            % Convert to matlab function
            obj.A_diff = matlabFunction(A_diff,'Vars',sym_vars); %#ok
            obj.B_diff = matlabFunction(B_diff,'Vars',sym_vars); %#ok
            obj.BwXw = matlabFunction(BwXw,'Vars',sym_vars);     %#ok
        end
        
        function dt = fw_approx(obj,x,u,theta,d) % Approximate total dynamics
            obj.approximate = true;
            dt = obj.fw(x,u,theta,d);
            obj.approximate = false;
        end
        
        function dt = f_approx(obj,x) % Approximate state dynamics
            obj.approximate = true;
            dt = obj.f_fcn(x);
            obj.approximate = false;
        end
        
        function dt = B_approx(obj,x) % Approximate input dynamics
            obj.approximate = true;
            dt = obj.B_fcn(x);
            obj.approximate = false;
        end
        
        function dt = G_approx(obj,varargin) % Approximate theta dynamics
            obj.approximate = true;
            dt = obj.G_fcn(varargin{:});
            obj.approximate = false;
        end
        
        function dt = E_approx(obj,x) % Approximate disturbance dynamics
            obj.approximate = true;
            dt = obj.E_fcn(x);
            obj.approximate = false;
        end
        
        function Y = sinx(obj,X)
            if obj.approximate % with Chebyshev polynomials
                Y = 0.9101*(X/(pi/3)) - 0.04466*(4*(X/(pi/3)).^3 - 3*(X/(pi/3)));
            else
                Y = sin(X);
            end
        end

        function Y = cosx(obj,X)
            if obj.approximate % with Chebyshev polynomials
                Y = 0.7441 - 0.2499*(2*(X/(pi/3)).^2 - 1);
            else
                Y = cos(X);
            end
        end
    end
    
    methods (Abstract)
        obj = initialize(obj)
        dt = f_fcn(obj,x)   % State dynamics
        dt = B_fcn(obj,x)   % Input dynamics
        dt = G_fcn(obj,x,u) % Theta dynamics
        dt = E_fcn(obj,x)   % Disturbance dynamics
    end
end
%% Main
% -------------------------------------------------------------------------
% File: main.m
% Author: Alexander Erdin (aerdin@ethz.ch)
% Date: 19th April 2024
% Modified: Mathieu Dubied (mdubied@ethz.ch)
% Date: 14th July 2024
% License: MIT
% Reference:
%
% -------------------------------------------------------------------------
clearvars;
close all;
clc;

% Choose system
sysName = 'MSDS';        % mass spring damper system
% sysName = 'Quadrotor';   % planar quadrotor

% Load parameters and create system
use_sos = false;

if strcmp(sysName, 'MSDS')
    fprintf('Offline routine for MSDS\n')
    params = PQRparamsMSDS();
    sys = MassSpringDamperSystem(params.dt, ...
                      'approximate',use_sos, ...
                      'param_uncertainty',params.param_uncertainty);    % TODO: check uncertainty
    W_idxs = [1,2];    % just as placeholders, will not be used
    monomial_degree = 0;    % constant CCM
    rho = 0.95;
else
    fprintf('Offline routine for Quadrotor \n')
    params = PQRparams();
    sys = PlanarQuadrotor(params.dt, ...
                      'integrator',params.integrator, ...
                      'approximate',use_sos, ...
                      'param_uncertainty',params.param_uncertainty);
    W_idxs = [3,4];  
    monomial_degree = 6;  
    rho = 0.95;
end
    
% Define gridding and recompute
n_tot = struct('rccm',5000,'rccm_check',1E4,'c_x',1E4,'c_u',1E4,'c_obs',1E4,'L_G',1E3,'G_M',1E3,'E_M',1E3,'w_bar',1E4,'delta_over',1E4,'M_under',1E4,'M_under_check',1E4);
recompute = struct('rccm',true,'c_x',true,'c_u',true,'c_obs',true,'L_G',true,'G_M',true,'E_M',true,'delta_over',true,'M_under',true);

% Create ccm class instance
ccm = CCM(sys,params, ...
          'use_sos', use_sos, ...
          'rho',rho, ...
          'monomial_degree',monomial_degree, ...
          'alpha', 0.6, ... % effective contraction rate >= alpha*rho
          'terminal_constraint',false, ...
          'n_tot', n_tot, ...
          'recompute',recompute, ...
          'W_idxs',W_idxs);  % Additional line for MSDS 
%%
% Store useful quantities in separate file, store it in folder
% "offline_constants"
store_file_name = sysName + "_offline_constants_" + ccm.rho + "_" + monomial_degree + ".mat";
path_to_save = fullfile('..','offline_constants',store_file_name);

if monomial_degree > 0
    M_under = ccm.M_under;
else
    M_under = 0;
end

% Offline constants common to all systems
rho = ccm.rho;
c_x = ccm.c_x;
c_u = ccm.c_u;
c_obs = ccm.c_obs;
L_G = ccm.L_G;
G_M = ccm.G_M;
E_M = ccm.E_M;
W_coef = ccm.W_coef;
Y_coef = ccm.Y_coef;
delta_inf = ccm.delta_inf;
delta_bar_0 = ccm.delta_bar_0;

F_u = params.F_u;
b_u = params.b_u;
F_x = params.F_x;
b_x = params.b_x;
d_max = params.disturbance_v;
obs_pos = params.obs_pos;

% Save system constants specific to either the MSDS or the Quadrotor
if strcmp(sysName, 'MSDS')
    m = ccm.sys.m;
    k = ccm.sys.k;
    c = ccm.sys.c;
    M = inv(W_coef);    % monomial degree is 0
    K = Y_coef*M;       % monomial degree is 0
    
    % Save file
    save(path_to_save, "M_under", "rho", "c_x", "c_u", "c_obs", ...
        "L_G", "G_M", "E_M", "M", "K", ...
        "delta_inf", "delta_bar_0", "monomial_degree", ...
        "F_u", "b_u", "F_x", "b_x", "d_max", "m", "k", "c");
else
    m = ccm.sys.m;
    J = ccm.sys.J;
    l = ccm.sys.l;
    
    % Save file
    save(path_to_save, "M_under", "rho", "c_x", "c_u", "c_obs", ...
        "L_G", "G_M", "E_M", "W_coef", "Y_coef", ...
        "delta_inf", "delta_bar_0","monomial_degree", ...
        "F_u", "b_u", "F_x", "b_x", "d_max", "m", "J", "l", "obs_pos")
end












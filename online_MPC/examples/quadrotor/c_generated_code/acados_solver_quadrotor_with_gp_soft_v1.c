/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "quadrotor_with_gp_soft_v1_model/quadrotor_with_gp_soft_v1_model.h"
#include "quadrotor_with_gp_soft_v1_constraints/quadrotor_with_gp_soft_v1_constraints.h"



#include "acados_solver_quadrotor_with_gp_soft_v1.h"

#define NX     QUADROTOR_WITH_GP_SOFT_V1_NX
#define NZ     QUADROTOR_WITH_GP_SOFT_V1_NZ
#define NU     QUADROTOR_WITH_GP_SOFT_V1_NU
#define NP     QUADROTOR_WITH_GP_SOFT_V1_NP
#define NY0    QUADROTOR_WITH_GP_SOFT_V1_NY0
#define NY     QUADROTOR_WITH_GP_SOFT_V1_NY
#define NYN    QUADROTOR_WITH_GP_SOFT_V1_NYN

#define NBX    QUADROTOR_WITH_GP_SOFT_V1_NBX
#define NBX0   QUADROTOR_WITH_GP_SOFT_V1_NBX0
#define NBU    QUADROTOR_WITH_GP_SOFT_V1_NBU
#define NG     QUADROTOR_WITH_GP_SOFT_V1_NG
#define NBXN   QUADROTOR_WITH_GP_SOFT_V1_NBXN
#define NGN    QUADROTOR_WITH_GP_SOFT_V1_NGN

#define NH     QUADROTOR_WITH_GP_SOFT_V1_NH
#define NHN    QUADROTOR_WITH_GP_SOFT_V1_NHN
#define NH0    QUADROTOR_WITH_GP_SOFT_V1_NH0
#define NPHI   QUADROTOR_WITH_GP_SOFT_V1_NPHI
#define NPHIN  QUADROTOR_WITH_GP_SOFT_V1_NPHIN
#define NPHI0  QUADROTOR_WITH_GP_SOFT_V1_NPHI0
#define NR     QUADROTOR_WITH_GP_SOFT_V1_NR

#define NS     QUADROTOR_WITH_GP_SOFT_V1_NS
#define NS0    QUADROTOR_WITH_GP_SOFT_V1_NS0
#define NSN    QUADROTOR_WITH_GP_SOFT_V1_NSN

#define NSBX   QUADROTOR_WITH_GP_SOFT_V1_NSBX
#define NSBU   QUADROTOR_WITH_GP_SOFT_V1_NSBU
#define NSH0   QUADROTOR_WITH_GP_SOFT_V1_NSH0
#define NSH    QUADROTOR_WITH_GP_SOFT_V1_NSH
#define NSHN   QUADROTOR_WITH_GP_SOFT_V1_NSHN
#define NSG    QUADROTOR_WITH_GP_SOFT_V1_NSG
#define NSPHI0 QUADROTOR_WITH_GP_SOFT_V1_NSPHI0
#define NSPHI  QUADROTOR_WITH_GP_SOFT_V1_NSPHI
#define NSPHIN QUADROTOR_WITH_GP_SOFT_V1_NSPHIN
#define NSGN   QUADROTOR_WITH_GP_SOFT_V1_NSGN
#define NSBXN  QUADROTOR_WITH_GP_SOFT_V1_NSBXN



// ** solver data **

quadrotor_with_gp_soft_v1_solver_capsule * quadrotor_with_gp_soft_v1_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(quadrotor_with_gp_soft_v1_solver_capsule));
    quadrotor_with_gp_soft_v1_solver_capsule *capsule = (quadrotor_with_gp_soft_v1_solver_capsule *) capsule_mem;

    return capsule;
}


int quadrotor_with_gp_soft_v1_acados_free_capsule(quadrotor_with_gp_soft_v1_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int quadrotor_with_gp_soft_v1_acados_create(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    int N_shooting_intervals = QUADROTOR_WITH_GP_SOFT_V1_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return quadrotor_with_gp_soft_v1_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int quadrotor_with_gp_soft_v1_acados_update_time_steps(quadrotor_with_gp_soft_v1_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "quadrotor_with_gp_soft_v1_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 1
 */
void quadrotor_with_gp_soft_v1_acados_create_1_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 2
 */
ocp_nlp_dims* quadrotor_with_gp_soft_v1_acados_create_2_create_and_set_dimensions(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 17
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    nbxe[0] = 0;
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 3
 */
void quadrotor_with_gp_soft_v1_acados_create_3_create_and_set_functions(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;


    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_param_casadi_create(&capsule->__CAPSULE_FNC__ , 76); \
    } while(false)
    MAP_CASADI_FNC(nl_constr_h_0_fun_jac, quadrotor_with_gp_soft_v1_constr_h_0_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_0_fun, quadrotor_with_gp_soft_v1_constr_h_0_fun);
    // constraints.constr_type == "BGH" and dims.nh > 0
    capsule->nl_constr_h_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac[i], quadrotor_with_gp_soft_v1_constr_h_fun_jac_uxt_zt);
    }
    capsule->nl_constr_h_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun[i], quadrotor_with_gp_soft_v1_constr_h_fun);
    }
    

    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, quadrotor_with_gp_soft_v1_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, quadrotor_with_gp_soft_v1_constr_h_e_fun);


    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun[i], quadrotor_with_gp_soft_v1_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], quadrotor_with_gp_soft_v1_dyn_disc_phi_fun_jac);
    }

#undef MAP_CASADI_FNC
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 4
 */
void quadrotor_with_gp_soft_v1_acados_create_4_set_default_parameters(quadrotor_with_gp_soft_v1_solver_capsule* capsule) {
    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        quadrotor_with_gp_soft_v1_acados_update_params(capsule, i, p, NP);
    }
    free(p);
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 5
 */
void quadrotor_with_gp_soft_v1_acados_create_5_set_nlp_in(quadrotor_with_gp_soft_v1_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps

    if (new_time_steps)
    {
        quadrotor_with_gp_soft_v1_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {double time_step = 0.025;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_step);
        }
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    yref_0[0] = 1;
    yref_0[1] = 2;
    yref_0[7] = 2.4;
    yref_0[8] = 2.4;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 20;
    W_0[1+(NY0) * 1] = 20;
    W_0[2+(NY0) * 2] = 10;
    W_0[3+(NY0) * 3] = 1;
    W_0[4+(NY0) * 4] = 1;
    W_0[5+(NY0) * 5] = 10;
    W_0[6+(NY0) * 6] = 0.1;
    W_0[7+(NY0) * 7] = 5;
    W_0[8+(NY0) * 8] = 5;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    Vx_0[2+(NY0) * 2] = 1;
    Vx_0[3+(NY0) * 3] = 1;
    Vx_0[4+(NY0) * 4] = 1;
    Vx_0[5+(NY0) * 5] = 1;
    Vx_0[6+(NY0) * 6] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[7+(NY0) * 0] = 1;
    Vu_0[8+(NY0) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:
    yref[0] = 1;
    yref[1] = 2;
    yref[7] = 2.4;
    yref[8] = 2.4;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 20;
    W[1+(NY) * 1] = 20;
    W[2+(NY) * 2] = 10;
    W[3+(NY) * 3] = 1;
    W[4+(NY) * 4] = 1;
    W[5+(NY) * 5] = 10;
    W[6+(NY) * 6] = 0.1;
    W[7+(NY) * 7] = 5;
    W[8+(NY) * 8] = 5;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    Vx[2+(NY) * 2] = 1;
    Vx[3+(NY) * 3] = 1;
    Vx[4+(NY) * 4] = 1;
    Vx[5+(NY) * 5] = 1;
    Vx[6+(NY) * 6] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    
    Vu[7+(NY) * 0] = 1;
    Vu[8+(NY) * 1] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    yref_e[0] = 1;
    yref_e[1] = 2;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 20;
    W_e[1+(NYN) * 1] = 20;
    W_e[2+(NYN) * 2] = 10;
    W_e[3+(NYN) * 3] = 1;
    W_e[4+(NYN) * 4] = 1;
    W_e[5+(NYN) * 5] = 10;
    W_e[6+(NYN) * 6] = 0.1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    Vx_e[2+(NYN) * 2] = 1;
    Vx_e[3+(NYN) * 3] = 1;
    Vx_e[4+(NYN) * 4] = 1;
    Vx_e[5+(NYN) * 5] = 1;
    Vx_e[6+(NYN) * 6] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);






    /**** Constraints ****/

    // bounds for initial stage



    // set up nonlinear constraints for last stage
    double* luh_0 = calloc(2*NH0, sizeof(double));
    double* lh_0 = luh_0;
    double* uh_0 = luh_0 + NH0;
    
    lh_0[9] = -1000000;
    lh_0[10] = -1000000;
    lh_0[11] = -1000000;
    lh_0[12] = -1000000;
    lh_0[13] = -1000000;
    lh_0[14] = -1000000;
    lh_0[15] = -1000000;
    lh_0[16] = -1000000;
    lh_0[17] = -1000000;
    lh_0[18] = -1000000;
    lh_0[19] = -1000000;
    lh_0[20] = -1000000;
    lh_0[21] = -1000000;
    lh_0[22] = -1000000;
    lh_0[23] = -1000000;
    lh_0[24] = -1000000;

    
    uh_0[0] = 1000000;
    uh_0[1] = 1000000;
    uh_0[9] = 20;
    uh_0[10] = 20;
    uh_0[11] = 20;
    uh_0[12] = 20;
    uh_0[13] = 0.39269908169872414;
    uh_0[14] = 1;
    uh_0[15] = 1;
    uh_0[16] = 1.0471975511965976;
    uh_0[17] = 0.39269908169872414;
    uh_0[18] = 1;
    uh_0[19] = 1;
    uh_0[20] = 1.0471975511965976;
    uh_0[21] = 3.5;
    uh_0[22] = 3.5;
    uh_0[23] = 1;
    uh_0[24] = 1;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun_jac", &capsule->nl_constr_h_0_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun", &capsule->nl_constr_h_0_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lh", lh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "uh", uh_0);
    free(luh_0);





    /* constraints that are the same for initial and intermediate */












    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;

    
    lh[0] = -1000000;
    lh[1] = -1000000;
    lh[2] = -1000000;
    lh[3] = -1000000;
    lh[4] = -1000000;
    lh[5] = -1000000;
    lh[6] = -1000000;
    lh[7] = -1000000;
    lh[8] = -1000000;
    lh[9] = -1000000;
    lh[10] = -1000000;
    lh[11] = -1000000;
    lh[12] = -1000000;
    lh[13] = -1000000;
    lh[14] = -1000000;
    lh[15] = -1000000;

    
    uh[0] = 20;
    uh[1] = 20;
    uh[2] = 20;
    uh[3] = 20;
    uh[4] = 0.39269908169872414;
    uh[5] = 1;
    uh[6] = 1;
    uh[7] = 1.0471975511965976;
    uh[8] = 0.39269908169872414;
    uh[9] = 1;
    uh[10] = 1;
    uh[11] = 1.0471975511965976;
    uh[12] = 3.5;
    uh[13] = 3.5;
    uh[14] = 1;
    uh[15] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uh", uh);
    }
    free(luh);



    /* terminal constraints */













    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*NHN, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + NHN;
    
    lh_e[0] = 1;
    lh_e[1] = 2;
    lh_e[7] = -1000000;
    lh_e[8] = -1000000;
    lh_e[9] = -1000000;
    lh_e[10] = -1000000;
    lh_e[11] = -1000000;
    lh_e[12] = -1000000;
    lh_e[13] = -1000000;
    lh_e[14] = -1000000;
    lh_e[15] = -1000000;
    lh_e[16] = -1000000;
    lh_e[17] = -1000000;
    lh_e[18] = -1000000;

    
    uh_e[0] = 1;
    uh_e[1] = 2;
    uh_e[6] = 0.6651830288367376;
    uh_e[7] = 20;
    uh_e[8] = 20;
    uh_e[9] = 20;
    uh_e[10] = 20;
    uh_e[11] = 0.39269908169872414;
    uh_e[12] = 1;
    uh_e[13] = 1;
    uh_e[14] = 1.0471975511965976;
    uh_e[15] = 0.39269908169872414;
    uh_e[16] = 1;
    uh_e[17] = 1;
    uh_e[18] = 1.0471975511965976;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uh", uh_e);
    free(luh_e);
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 6
 */
void quadrotor_with_gp_soft_v1_acados_create_6_set_opts(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/

int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization", "merit_backtracking");

    double alpha_min = 0.05;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "alpha_min", &alpha_min);

    double alpha_reduction = 0.7;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "alpha_reduction", &alpha_reduction);

    int line_search_use_sufficient_descent = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "line_search_use_sufficient_descent", &line_search_use_sufficient_descent);

    int globalization_use_SOC = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_use_SOC", &globalization_use_SOC);

    double eps_sufficient_descent = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "eps_sufficient_descent", &eps_sufficient_descent);int full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "full_step_dual", &full_step_dual);

    double nlp_solver_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "step_length", &nlp_solver_step_length);

    double levenberg_marquardt = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;
    // NOTE: there is no condensing happening here!
    qp_solver_cond_N = N;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");


    // set SQP specific options
    double nlp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 200;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    int initialize_t_slacks = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "initialize_t_slacks", &initialize_t_slacks);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 7
 */
void quadrotor_with_gp_soft_v1_acados_create_7_set_nlp_out(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with zeros

    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 8
 */
//void quadrotor_with_gp_soft_v1_acados_create_8_create_solver(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
//{
//    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
//}

/**
 * Internal function for quadrotor_with_gp_soft_v1_acados_create: step 9
 */
int quadrotor_with_gp_soft_v1_acados_create_9_precompute(quadrotor_with_gp_soft_v1_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int quadrotor_with_gp_soft_v1_acados_create_with_discretization(quadrotor_with_gp_soft_v1_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != QUADROTOR_WITH_GP_SOFT_V1_N && !new_time_steps) {
        fprintf(stderr, "quadrotor_with_gp_soft_v1_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, QUADROTOR_WITH_GP_SOFT_V1_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    quadrotor_with_gp_soft_v1_acados_create_1_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 3) create and set dimensions
    capsule->nlp_dims = quadrotor_with_gp_soft_v1_acados_create_2_create_and_set_dimensions(capsule);
    quadrotor_with_gp_soft_v1_acados_create_3_create_and_set_functions(capsule);

    // 4) set default parameters in functions
    quadrotor_with_gp_soft_v1_acados_create_4_set_default_parameters(capsule);

    // 5) create and set nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_with_gp_soft_v1_acados_create_5_set_nlp_in(capsule, N, new_time_steps);

    // 6) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_with_gp_soft_v1_acados_create_6_set_opts(capsule);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_with_gp_soft_v1_acados_create_7_set_nlp_out(capsule);

    // 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
    //quadrotor_with_gp_soft_v1_acados_create_8_create_solver(capsule);

    // 9) do precomputations
    int status = quadrotor_with_gp_soft_v1_acados_create_9_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int quadrotor_with_gp_soft_v1_acados_update_qp_solver_cond_N(quadrotor_with_gp_soft_v1_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from quadrotor_with_gp_soft_v1_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);

    // -> 9) do precomputations
    int status = quadrotor_with_gp_soft_v1_acados_create_9_precompute(capsule);
    return status;
}


int quadrotor_with_gp_soft_v1_acados_reset(quadrotor_with_gp_soft_v1_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "t", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int quadrotor_with_gp_soft_v1_acados_update_params(quadrotor_with_gp_soft_v1_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 76;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

    const int N = capsule->nlp_solver_plan->N;
    if (stage < N && stage >= 0)
    {
        capsule->discr_dyn_phi_fun[stage].set_param(capsule->discr_dyn_phi_fun+stage, p);
        capsule->discr_dyn_phi_fun_jac_ut_xt[stage].set_param(capsule->discr_dyn_phi_fun_jac_ut_xt+stage, p);

        // constraints
        if (stage == 0)
        {
            capsule->nl_constr_h_0_fun_jac.set_param(&capsule->nl_constr_h_0_fun_jac, p);
            capsule->nl_constr_h_0_fun.set_param(&capsule->nl_constr_h_0_fun, p);
        }
        else
        {
            capsule->nl_constr_h_fun_jac[stage-1].set_param(capsule->nl_constr_h_fun_jac+stage-1, p);
            capsule->nl_constr_h_fun[stage-1].set_param(capsule->nl_constr_h_fun+stage-1, p);
        }

        // cost
        if (stage == 0)
        {
        }
        else // 0 < stage < N
        {
        }
    }

    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
        // constraints
        capsule->nl_constr_h_e_fun_jac.set_param(&capsule->nl_constr_h_e_fun_jac, p);
        capsule->nl_constr_h_e_fun.set_param(&capsule->nl_constr_h_e_fun, p);
    }

    return solver_status;
}


int quadrotor_with_gp_soft_v1_acados_update_params_sparse(quadrotor_with_gp_soft_v1_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    int solver_status = 0;

    int casadi_np = 76;
    if (casadi_np < n_update) {
        printf("quadrotor_with_gp_soft_v1_acados_update_params_sparse: trying to set %d parameters for external functions."
            " External function has %d parameters. Exiting.\n", n_update, casadi_np);
        exit(1);
    }
    // for (int i = 0; i < n_update; i++)
    // {
    //     if (idx[i] > casadi_np) {
    //         printf("quadrotor_with_gp_soft_v1_acados_update_params_sparse: attempt to set parameters with index %d, while"
    //             " external functions only has %d parameters. Exiting.\n", idx[i], casadi_np);
    //         exit(1);
    //     }
    //     printf("param %d value %e\n", idx[i], p[i]);
    // }
    const int N = capsule->nlp_solver_plan->N;
    if (stage < N && stage >= 0)
    {
        capsule->discr_dyn_phi_fun[stage].set_param_sparse(capsule->discr_dyn_phi_fun+stage, n_update, idx, p);
        capsule->discr_dyn_phi_fun_jac_ut_xt[stage].set_param_sparse(capsule->discr_dyn_phi_fun_jac_ut_xt+stage, n_update, idx, p);

        // cost & constraints
        if (stage == 0)
        {
            // cost
            // constraints
        
            capsule->nl_constr_h_0_fun_jac.set_param_sparse(&capsule->nl_constr_h_0_fun_jac, n_update, idx, p);
            capsule->nl_constr_h_0_fun.set_param_sparse(&capsule->nl_constr_h_0_fun, n_update, idx, p);
        
        }
        else // 0 < stage < N
        {

        
            capsule->nl_constr_h_fun_jac[stage-1].set_param_sparse(capsule->nl_constr_h_fun_jac+stage-1, n_update, idx, p);
            capsule->nl_constr_h_fun[stage-1].set_param_sparse(capsule->nl_constr_h_fun+stage-1, n_update, idx, p);
        }
    }

    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
        // constraints
    
        capsule->nl_constr_h_e_fun_jac.set_param_sparse(&capsule->nl_constr_h_e_fun_jac, n_update, idx, p);
        capsule->nl_constr_h_e_fun.set_param_sparse(&capsule->nl_constr_h_e_fun, n_update, idx, p);
    
    }


    return solver_status;
}

int quadrotor_with_gp_soft_v1_acados_solve(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int quadrotor_with_gp_soft_v1_acados_free(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);

    // cost

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    external_function_param_casadi_free(&capsule->nl_constr_h_0_fun_jac);
    external_function_param_casadi_free(&capsule->nl_constr_h_0_fun);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun);

    return 0;
}


void quadrotor_with_gp_soft_v1_acados_print_stats(quadrotor_with_gp_soft_v1_solver_capsule* capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    
    double stat[2400];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");

    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }

}

int quadrotor_with_gp_soft_v1_acados_custom_update(quadrotor_with_gp_soft_v1_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *quadrotor_with_gp_soft_v1_acados_get_nlp_in(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *quadrotor_with_gp_soft_v1_acados_get_nlp_out(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *quadrotor_with_gp_soft_v1_acados_get_sens_out(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *quadrotor_with_gp_soft_v1_acados_get_nlp_solver(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *quadrotor_with_gp_soft_v1_acados_get_nlp_config(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_config; }
void *quadrotor_with_gp_soft_v1_acados_get_nlp_opts(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *quadrotor_with_gp_soft_v1_acados_get_nlp_dims(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *quadrotor_with_gp_soft_v1_acados_get_nlp_plan(quadrotor_with_gp_soft_v1_solver_capsule* capsule) { return capsule->nlp_solver_plan; }

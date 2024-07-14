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
#include "quadrotor_RAMPC_v1_model/quadrotor_RAMPC_v1_model.h"
#include "quadrotor_RAMPC_v1_constraints/quadrotor_RAMPC_v1_constraints.h"



#include "acados_solver_quadrotor_RAMPC_v1.h"

#define NX     QUADROTOR_RAMPC_V1_NX
#define NZ     QUADROTOR_RAMPC_V1_NZ
#define NU     QUADROTOR_RAMPC_V1_NU
#define NP     QUADROTOR_RAMPC_V1_NP
#define NY0    QUADROTOR_RAMPC_V1_NY0
#define NY     QUADROTOR_RAMPC_V1_NY
#define NYN    QUADROTOR_RAMPC_V1_NYN

#define NBX    QUADROTOR_RAMPC_V1_NBX
#define NBX0   QUADROTOR_RAMPC_V1_NBX0
#define NBU    QUADROTOR_RAMPC_V1_NBU
#define NG     QUADROTOR_RAMPC_V1_NG
#define NBXN   QUADROTOR_RAMPC_V1_NBXN
#define NGN    QUADROTOR_RAMPC_V1_NGN

#define NH     QUADROTOR_RAMPC_V1_NH
#define NHN    QUADROTOR_RAMPC_V1_NHN
#define NH0    QUADROTOR_RAMPC_V1_NH0
#define NPHI   QUADROTOR_RAMPC_V1_NPHI
#define NPHIN  QUADROTOR_RAMPC_V1_NPHIN
#define NPHI0  QUADROTOR_RAMPC_V1_NPHI0
#define NR     QUADROTOR_RAMPC_V1_NR

#define NS     QUADROTOR_RAMPC_V1_NS
#define NS0    QUADROTOR_RAMPC_V1_NS0
#define NSN    QUADROTOR_RAMPC_V1_NSN

#define NSBX   QUADROTOR_RAMPC_V1_NSBX
#define NSBU   QUADROTOR_RAMPC_V1_NSBU
#define NSH0   QUADROTOR_RAMPC_V1_NSH0
#define NSH    QUADROTOR_RAMPC_V1_NSH
#define NSHN   QUADROTOR_RAMPC_V1_NSHN
#define NSG    QUADROTOR_RAMPC_V1_NSG
#define NSPHI0 QUADROTOR_RAMPC_V1_NSPHI0
#define NSPHI  QUADROTOR_RAMPC_V1_NSPHI
#define NSPHIN QUADROTOR_RAMPC_V1_NSPHIN
#define NSGN   QUADROTOR_RAMPC_V1_NSGN
#define NSBXN  QUADROTOR_RAMPC_V1_NSBXN



// ** solver data **

quadrotor_RAMPC_v1_solver_capsule * quadrotor_RAMPC_v1_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(quadrotor_RAMPC_v1_solver_capsule));
    quadrotor_RAMPC_v1_solver_capsule *capsule = (quadrotor_RAMPC_v1_solver_capsule *) capsule_mem;

    return capsule;
}


int quadrotor_RAMPC_v1_acados_free_capsule(quadrotor_RAMPC_v1_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int quadrotor_RAMPC_v1_acados_create(quadrotor_RAMPC_v1_solver_capsule* capsule)
{
    int N_shooting_intervals = QUADROTOR_RAMPC_V1_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return quadrotor_RAMPC_v1_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int quadrotor_RAMPC_v1_acados_update_time_steps(quadrotor_RAMPC_v1_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "quadrotor_RAMPC_v1_acados_update_time_steps: given number of time steps (= %d) " \
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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 1
 */
void quadrotor_RAMPC_v1_acados_create_1_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 2
 */
ocp_nlp_dims* quadrotor_RAMPC_v1_acados_create_2_create_and_set_dimensions(quadrotor_RAMPC_v1_solver_capsule* capsule)
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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 3
 */
void quadrotor_RAMPC_v1_acados_create_3_create_and_set_functions(quadrotor_RAMPC_v1_solver_capsule* capsule)
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
        external_function_param_casadi_create(&capsule->__CAPSULE_FNC__ , 132); \
    } while(false)
    MAP_CASADI_FNC(nl_constr_h_0_fun_jac, quadrotor_RAMPC_v1_constr_h_0_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_0_fun, quadrotor_RAMPC_v1_constr_h_0_fun);
    // constraints.constr_type == "BGH" and dims.nh > 0
    capsule->nl_constr_h_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac[i], quadrotor_RAMPC_v1_constr_h_fun_jac_uxt_zt);
    }
    capsule->nl_constr_h_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun[i], quadrotor_RAMPC_v1_constr_h_fun);
    }
    

    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, quadrotor_RAMPC_v1_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, quadrotor_RAMPC_v1_constr_h_e_fun);


    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun[i], quadrotor_RAMPC_v1_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], quadrotor_RAMPC_v1_dyn_disc_phi_fun_jac);
    }

#undef MAP_CASADI_FNC
}


/**
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 4
 */
void quadrotor_RAMPC_v1_acados_create_4_set_default_parameters(quadrotor_RAMPC_v1_solver_capsule* capsule) {
    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        quadrotor_RAMPC_v1_acados_update_params(capsule, i, p, NP);
    }
    free(p);
}


/**
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 5
 */
void quadrotor_RAMPC_v1_acados_create_5_set_nlp_in(quadrotor_RAMPC_v1_solver_capsule* capsule, const int N, double* new_time_steps)
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
        quadrotor_RAMPC_v1_acados_update_time_steps(capsule, N, new_time_steps);
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
    W_0[3+(NY0) * 3] = 5;
    W_0[4+(NY0) * 4] = 5;
    W_0[5+(NY0) * 5] = 10;
    W_0[6+(NY0) * 6] = 0.1;
    W_0[7+(NY0) * 7] = 5;
    W_0[8+(NY0) * 8] = 5;
    W_0[9+(NY0) * 9] = 0.1;
    W_0[10+(NY0) * 10] = 0.1;
    W_0[11+(NY0) * 11] = 0.1;
    W_0[12+(NY0) * 12] = 0.1;
    W_0[13+(NY0) * 13] = 0.1;
    W_0[14+(NY0) * 14] = 0.1;
    W_0[15+(NY0) * 15] = 0.1;
    W_0[16+(NY0) * 16] = 0.1;
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
    Vu_0[9+(NY0) * 2] = 1;
    Vu_0[10+(NY0) * 3] = 1;
    Vu_0[11+(NY0) * 4] = 1;
    Vu_0[12+(NY0) * 5] = 1;
    Vu_0[13+(NY0) * 6] = 1;
    Vu_0[14+(NY0) * 7] = 1;
    Vu_0[15+(NY0) * 8] = 1;
    Vu_0[16+(NY0) * 9] = 1;
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
    W[3+(NY) * 3] = 5;
    W[4+(NY) * 4] = 5;
    W[5+(NY) * 5] = 10;
    W[6+(NY) * 6] = 0.1;
    W[7+(NY) * 7] = 5;
    W[8+(NY) * 8] = 5;
    W[9+(NY) * 9] = 0.1;
    W[10+(NY) * 10] = 0.1;
    W[11+(NY) * 11] = 0.1;
    W[12+(NY) * 12] = 0.1;
    W[13+(NY) * 13] = 0.1;
    W[14+(NY) * 14] = 0.1;
    W[15+(NY) * 15] = 0.1;
    W[16+(NY) * 16] = 0.1;

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
    Vu[9+(NY) * 2] = 1;
    Vu[10+(NY) * 3] = 1;
    Vu[11+(NY) * 4] = 1;
    Vu[12+(NY) * 5] = 1;
    Vu[13+(NY) * 6] = 1;
    Vu[14+(NY) * 7] = 1;
    Vu[15+(NY) * 8] = 1;
    Vu[16+(NY) * 9] = 1;

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
    W_e[0+(NYN) * 0] = 200;
    W_e[1+(NYN) * 1] = 200;
    W_e[2+(NYN) * 2] = 100;
    W_e[3+(NYN) * 3] = 50;
    W_e[4+(NYN) * 4] = 50;
    W_e[5+(NYN) * 5] = 100;
    W_e[6+(NYN) * 6] = 1;
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



    // slacks initial
    double* zlu0_mem = calloc(4*NS0, sizeof(double));
    double* Zl_0 = zlu0_mem+NS0*0;
    double* Zu_0 = zlu0_mem+NS0*1;
    double* zl_0 = zlu0_mem+NS0*2;
    double* zu_0 = zlu0_mem+NS0*3;

    // change only the non-zero elements:
    

    

    
    zl_0[0] = 10000;
    zl_0[1] = 10000;
    zl_0[2] = 10000;
    zl_0[3] = 10000;
    zl_0[4] = 10000;
    zl_0[5] = 10000;
    zl_0[6] = 10000;
    zl_0[7] = 10000;
    zl_0[8] = 10000;
    zl_0[9] = 10000;
    zl_0[10] = 10000;
    zl_0[11] = 10000;
    zl_0[12] = 10000;
    zl_0[13] = 10000;
    zl_0[14] = 10000;
    zl_0[15] = 10000;
    zl_0[16] = 10000;
    zl_0[17] = 10000;
    zl_0[18] = 10000;
    zl_0[19] = 10000;
    zl_0[20] = 10000;
    zl_0[21] = 10000;
    zl_0[22] = 10000;
    zl_0[23] = 10000;
    zl_0[24] = 10000;
    zl_0[25] = 10000;
    zl_0[26] = 10000;
    zl_0[27] = 10000;
    zl_0[28] = 10000;
    zl_0[29] = 10000;
    zl_0[30] = 10000;
    zl_0[31] = 10000;

    
    zu_0[0] = 10000;
    zu_0[1] = 10000;
    zu_0[2] = 10000;
    zu_0[3] = 10000;
    zu_0[4] = 10000;
    zu_0[5] = 10000;
    zu_0[6] = 10000;
    zu_0[7] = 10000;
    zu_0[8] = 10000;
    zu_0[9] = 10000;
    zu_0[10] = 10000;
    zu_0[11] = 10000;
    zu_0[12] = 10000;
    zu_0[13] = 10000;
    zu_0[14] = 10000;
    zu_0[15] = 10000;
    zu_0[16] = 10000;
    zu_0[17] = 10000;
    zu_0[18] = 10000;
    zu_0[19] = 10000;
    zu_0[20] = 10000;
    zu_0[21] = 10000;
    zu_0[22] = 10000;
    zu_0[23] = 10000;
    zu_0[24] = 10000;
    zu_0[25] = 10000;
    zu_0[26] = 10000;
    zu_0[27] = 10000;
    zu_0[28] = 10000;
    zu_0[29] = 10000;
    zu_0[30] = 10000;
    zu_0[31] = 10000;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zl", Zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Zu", Zu_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zl", zl_0);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "zu", zu_0);
    free(zlu0_mem);
    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    zl[0] = 10000;
    zl[1] = 10000;
    zl[2] = 10000;
    zl[3] = 10000;
    zl[4] = 10000;
    zl[5] = 10000;
    zl[6] = 10000;
    zl[7] = 10000;
    zl[8] = 10000;
    zl[9] = 10000;
    zl[10] = 10000;
    zl[11] = 10000;
    zl[12] = 10000;
    zl[13] = 10000;
    zl[14] = 10000;
    zl[15] = 10000;
    zl[16] = 10000;
    zl[17] = 10000;
    zl[18] = 10000;
    zl[19] = 10000;
    zl[20] = 10000;
    zl[21] = 10000;
    zl[22] = 10000;
    zl[23] = 10000;
    zu[0] = 10000;
    zu[1] = 10000;
    zu[2] = 10000;
    zu[3] = 10000;
    zu[4] = 10000;
    zu[5] = 10000;
    zu[6] = 10000;
    zu[7] = 10000;
    zu[8] = 10000;
    zu[9] = 10000;
    zu[10] = 10000;
    zu[11] = 10000;
    zu[12] = 10000;
    zu[13] = 10000;
    zu[14] = 10000;
    zu[15] = 10000;
    zu[16] = 10000;
    zu[17] = 10000;
    zu[18] = 10000;
    zu[19] = 10000;
    zu[20] = 10000;
    zu[21] = 10000;
    zu[22] = 10000;
    zu[23] = 10000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);


    // slacks terminal
    double* zluemem = calloc(4*NSN, sizeof(double));
    double* Zl_e = zluemem+NSN*0;
    double* Zu_e = zluemem+NSN*1;
    double* zl_e = zluemem+NSN*2;
    double* zu_e = zluemem+NSN*3;

    // change only the non-zero elements:
    

    

    
    zl_e[0] = 10000;
    zl_e[1] = 10000;
    zl_e[2] = 10000;
    zl_e[3] = 10000;
    zl_e[4] = 10000;
    zl_e[5] = 10000;
    zl_e[6] = 10000;
    zl_e[7] = 10000;
    zl_e[8] = 10000;
    zl_e[9] = 10000;
    zl_e[10] = 10000;
    zl_e[11] = 10000;
    zl_e[12] = 10000;
    zl_e[13] = 10000;
    zl_e[14] = 10000;
    zl_e[15] = 10000;
    zl_e[16] = 10000;
    zl_e[17] = 10000;
    zl_e[18] = 10000;

    
    zu_e[0] = 10000;
    zu_e[1] = 10000;
    zu_e[2] = 10000;
    zu_e[3] = 10000;
    zu_e[4] = 10000;
    zu_e[5] = 10000;
    zu_e[6] = 10000;
    zu_e[7] = 10000;
    zu_e[8] = 10000;
    zu_e[9] = 10000;
    zu_e[10] = 10000;
    zu_e[11] = 10000;
    zu_e[12] = 10000;
    zu_e[13] = 10000;
    zu_e[14] = 10000;
    zu_e[15] = 10000;
    zu_e[16] = 10000;
    zu_e[17] = 10000;
    zu_e[18] = 10000;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
    free(zluemem);

    /**** Constraints ****/

    // bounds for initial stage



    // set up nonlinear constraints for last stage
    double* luh_0 = calloc(2*NH0, sizeof(double));
    double* lh_0 = luh_0;
    double* uh_0 = luh_0 + NH0;
    
    lh_0[8] = -1000000;
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
    lh_0[24] = 1;
    lh_0[25] = 1;
    lh_0[26] = 1;
    lh_0[27] = 1;
    lh_0[28] = 1;
    lh_0[29] = 1;
    lh_0[30] = 1;
    lh_0[31] = 1;

    
    uh_0[0] = 1000000;
    uh_0[1] = 1000000;
    uh_0[8] = 20;
    uh_0[9] = 20;
    uh_0[10] = 20;
    uh_0[11] = 20;
    uh_0[12] = 0.39269908169872414;
    uh_0[13] = 1;
    uh_0[14] = 1;
    uh_0[15] = 1.0471975511965976;
    uh_0[16] = 0.39269908169872414;
    uh_0[17] = 1;
    uh_0[18] = 1;
    uh_0[19] = 1.0471975511965976;
    uh_0[20] = 3.5;
    uh_0[21] = 3.5;
    uh_0[22] = 1;
    uh_0[23] = 1;
    uh_0[24] = 1;
    uh_0[25] = 1;
    uh_0[26] = 1;
    uh_0[27] = 1;
    uh_0[28] = 1;
    uh_0[29] = 1;
    uh_0[30] = 1;
    uh_0[31] = 1;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun_jac", &capsule->nl_constr_h_0_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "nl_constr_h_fun", &capsule->nl_constr_h_0_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lh", lh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "uh", uh_0);
    free(luh_0);


    // set up soft bounds for nonlinear constraints
    int* idxsh_0 = malloc(NSH0 * sizeof(int));
    
    idxsh_0[0] = 0;
    idxsh_0[1] = 1;
    idxsh_0[2] = 2;
    idxsh_0[3] = 3;
    idxsh_0[4] = 4;
    idxsh_0[5] = 5;
    idxsh_0[6] = 6;
    idxsh_0[7] = 7;
    idxsh_0[8] = 8;
    idxsh_0[9] = 9;
    idxsh_0[10] = 10;
    idxsh_0[11] = 11;
    idxsh_0[12] = 12;
    idxsh_0[13] = 13;
    idxsh_0[14] = 14;
    idxsh_0[15] = 15;
    idxsh_0[16] = 16;
    idxsh_0[17] = 17;
    idxsh_0[18] = 18;
    idxsh_0[19] = 19;
    idxsh_0[20] = 20;
    idxsh_0[21] = 21;
    idxsh_0[22] = 22;
    idxsh_0[23] = 23;
    idxsh_0[24] = 24;
    idxsh_0[25] = 25;
    idxsh_0[26] = 26;
    idxsh_0[27] = 27;
    idxsh_0[28] = 28;
    idxsh_0[29] = 29;
    idxsh_0[30] = 30;
    idxsh_0[31] = 31;
    double* lush_0 = calloc(2*NSH0, sizeof(double));
    double* lsh_0 = lush_0;
    double* ush_0 = lush_0 + NSH0;
    

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxsh", idxsh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lsh", lsh_0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ush", ush_0);
    free(idxsh_0);
    free(lush_0);



    /* constraints that are the same for initial and intermediate */




    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    
    idxsh[0] = 0;
    idxsh[1] = 1;
    idxsh[2] = 2;
    idxsh[3] = 3;
    idxsh[4] = 4;
    idxsh[5] = 5;
    idxsh[6] = 6;
    idxsh[7] = 7;
    idxsh[8] = 8;
    idxsh[9] = 9;
    idxsh[10] = 10;
    idxsh[11] = 11;
    idxsh[12] = 12;
    idxsh[13] = 13;
    idxsh[14] = 14;
    idxsh[15] = 15;
    idxsh[16] = 16;
    idxsh[17] = 17;
    idxsh[18] = 18;
    idxsh[19] = 19;
    idxsh[20] = 20;
    idxsh[21] = 21;
    idxsh[22] = 22;
    idxsh[23] = 23;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;
    

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ush", ush);
    }
    free(idxsh);
    free(lush);








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
    lh[16] = 1;
    lh[17] = 1;
    lh[18] = 1;
    lh[19] = 1;
    lh[20] = 1;
    lh[21] = 1;
    lh[22] = 1;
    lh[23] = 1;

    
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
    uh[16] = 1;
    uh[17] = 1;
    uh[18] = 1;
    uh[19] = 1;
    uh[20] = 1;
    uh[21] = 1;
    uh[22] = 1;
    uh[23] = 1;

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





    // set up soft bounds for nonlinear constraints
    int* idxsh_e = malloc(NSHN * sizeof(int));
    
    idxsh_e[0] = 0;
    idxsh_e[1] = 1;
    idxsh_e[2] = 2;
    idxsh_e[3] = 3;
    idxsh_e[4] = 4;
    idxsh_e[5] = 5;
    idxsh_e[6] = 6;
    idxsh_e[7] = 7;
    idxsh_e[8] = 8;
    idxsh_e[9] = 9;
    idxsh_e[10] = 10;
    idxsh_e[11] = 11;
    idxsh_e[12] = 12;
    idxsh_e[13] = 13;
    idxsh_e[14] = 14;
    idxsh_e[15] = 15;
    idxsh_e[16] = 16;
    idxsh_e[17] = 17;
    idxsh_e[18] = 18;
    double* lush_e = calloc(2*NSHN, sizeof(double));
    double* lsh_e = lush_e;
    double* ush_e = lush_e + NSHN;
    

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ush", ush_e);
    free(idxsh_e);
    free(lush_e);








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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 6
 */
void quadrotor_RAMPC_v1_acados_create_6_set_opts(quadrotor_RAMPC_v1_solver_capsule* capsule)
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

    int nlp_solver_max_iter = 100;
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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 7
 */
void quadrotor_RAMPC_v1_acados_create_7_set_nlp_out(quadrotor_RAMPC_v1_solver_capsule* capsule)
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
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 8
 */
//void quadrotor_RAMPC_v1_acados_create_8_create_solver(quadrotor_RAMPC_v1_solver_capsule* capsule)
//{
//    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
//}

/**
 * Internal function for quadrotor_RAMPC_v1_acados_create: step 9
 */
int quadrotor_RAMPC_v1_acados_create_9_precompute(quadrotor_RAMPC_v1_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int quadrotor_RAMPC_v1_acados_create_with_discretization(quadrotor_RAMPC_v1_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != QUADROTOR_RAMPC_V1_N && !new_time_steps) {
        fprintf(stderr, "quadrotor_RAMPC_v1_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, QUADROTOR_RAMPC_V1_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    quadrotor_RAMPC_v1_acados_create_1_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 3) create and set dimensions
    capsule->nlp_dims = quadrotor_RAMPC_v1_acados_create_2_create_and_set_dimensions(capsule);
    quadrotor_RAMPC_v1_acados_create_3_create_and_set_functions(capsule);

    // 4) set default parameters in functions
    quadrotor_RAMPC_v1_acados_create_4_set_default_parameters(capsule);

    // 5) create and set nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_RAMPC_v1_acados_create_5_set_nlp_in(capsule, N, new_time_steps);

    // 6) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_RAMPC_v1_acados_create_6_set_opts(capsule);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    quadrotor_RAMPC_v1_acados_create_7_set_nlp_out(capsule);

    // 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);
    //quadrotor_RAMPC_v1_acados_create_8_create_solver(capsule);

    // 9) do precomputations
    int status = quadrotor_RAMPC_v1_acados_create_9_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int quadrotor_RAMPC_v1_acados_update_qp_solver_cond_N(quadrotor_RAMPC_v1_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from quadrotor_RAMPC_v1_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts);

    // -> 9) do precomputations
    int status = quadrotor_RAMPC_v1_acados_create_9_precompute(capsule);
    return status;
}


int quadrotor_RAMPC_v1_acados_reset(quadrotor_RAMPC_v1_solver_capsule* capsule, int reset_qp_solver_mem)
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




int quadrotor_RAMPC_v1_acados_update_params(quadrotor_RAMPC_v1_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 132;
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


int quadrotor_RAMPC_v1_acados_update_params_sparse(quadrotor_RAMPC_v1_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    int solver_status = 0;

    int casadi_np = 132;
    if (casadi_np < n_update) {
        printf("quadrotor_RAMPC_v1_acados_update_params_sparse: trying to set %d parameters for external functions."
            " External function has %d parameters. Exiting.\n", n_update, casadi_np);
        exit(1);
    }
    // for (int i = 0; i < n_update; i++)
    // {
    //     if (idx[i] > casadi_np) {
    //         printf("quadrotor_RAMPC_v1_acados_update_params_sparse: attempt to set parameters with index %d, while"
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

int quadrotor_RAMPC_v1_acados_solve(quadrotor_RAMPC_v1_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int quadrotor_RAMPC_v1_acados_free(quadrotor_RAMPC_v1_solver_capsule* capsule)
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


void quadrotor_RAMPC_v1_acados_print_stats(quadrotor_RAMPC_v1_solver_capsule* capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    
    double stat[1200];
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

int quadrotor_RAMPC_v1_acados_custom_update(quadrotor_RAMPC_v1_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *quadrotor_RAMPC_v1_acados_get_nlp_in(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *quadrotor_RAMPC_v1_acados_get_nlp_out(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *quadrotor_RAMPC_v1_acados_get_sens_out(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *quadrotor_RAMPC_v1_acados_get_nlp_solver(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *quadrotor_RAMPC_v1_acados_get_nlp_config(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_config; }
void *quadrotor_RAMPC_v1_acados_get_nlp_opts(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *quadrotor_RAMPC_v1_acados_get_nlp_dims(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *quadrotor_RAMPC_v1_acados_get_nlp_plan(quadrotor_RAMPC_v1_solver_capsule* capsule) { return capsule->nlp_solver_plan; }

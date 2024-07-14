/* This file was automatically generated by CasADi 3.6.4.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quadrotor_RAMPC_v2_constr_h_0_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[14] = {10, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[136] = {132, 1, 0, 132, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131};
static const casadi_int casadi_s4[39] = {35, 1, 0, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34};

/* quadrotor_RAMPC_v2_constr_h_0_fun:(i0[7],i1[10],i2[],i3[132])->(o0[35]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][6] : 0;
  a1=casadi_sq(a0);
  a2=1.0359384638773170e+01;
  a3=arg[3]? arg[3][126] : 0;
  a4=arg[0]? arg[0][0] : 0;
  a5=(a3-a4);
  a2=(a2*a5);
  a6=2.5783435321485895e-07;
  a7=arg[3]? arg[3][127] : 0;
  a8=arg[0]? arg[0][1] : 0;
  a9=(a7-a8);
  a10=(a6*a9);
  a2=(a2+a10);
  a10=-8.8798781921928320e+00;
  a11=arg[3]? arg[3][128] : 0;
  a12=arg[0]? arg[0][2] : 0;
  a13=(a11-a12);
  a14=(a10*a13);
  a2=(a2+a14);
  a14=6.7556718886458063e+00;
  a15=arg[3]? arg[3][129] : 0;
  a16=arg[0]? arg[0][3] : 0;
  a17=(a15-a16);
  a18=(a14*a17);
  a2=(a2+a18);
  a18=9.5845032274525965e-08;
  a19=arg[3]? arg[3][130] : 0;
  a20=arg[0]? arg[0][4] : 0;
  a21=(a19-a20);
  a22=(a18*a21);
  a2=(a2+a22);
  a22=-4.6154723887620741e-01;
  a23=arg[3]? arg[3][131] : 0;
  a24=arg[0]? arg[0][5] : 0;
  a25=(a23-a24);
  a26=(a22*a25);
  a2=(a2+a26);
  a26=(a3-a4);
  a2=(a2*a26);
  a6=(a6*a5);
  a26=3.3976831598989974e+00;
  a26=(a26*a9);
  a6=(a6+a26);
  a26=5.6999629708142419e-08;
  a27=(a26*a13);
  a6=(a6+a27);
  a27=2.0658779001895284e-07;
  a28=(a27*a17);
  a6=(a6+a28);
  a28=1.7366572829657221e-07;
  a29=(a28*a21);
  a6=(a6+a29);
  a29=1.6747722305979081e-07;
  a30=(a29*a25);
  a6=(a6+a30);
  a30=(a7-a8);
  a6=(a6*a30);
  a2=(a2+a6);
  a10=(a10*a5);
  a26=(a26*a9);
  a10=(a10+a26);
  a26=2.7915158272898665e+01;
  a26=(a26*a13);
  a10=(a10+a26);
  a26=-9.5844271533744294e+00;
  a6=(a26*a17);
  a10=(a10+a6);
  a6=6.6584443711750246e-08;
  a30=(a6*a21);
  a10=(a10+a30);
  a30=1.3148404173059873e+00;
  a31=(a30*a25);
  a10=(a10+a31);
  a31=(a11-a12);
  a10=(a10*a31);
  a2=(a2+a10);
  a14=(a14*a5);
  a27=(a27*a9);
  a14=(a14+a27);
  a26=(a26*a13);
  a14=(a14+a26);
  a26=6.6182203696040682e+00;
  a26=(a26*a17);
  a14=(a14+a26);
  a26=8.5171020573893735e-08;
  a27=(a26*a21);
  a14=(a14+a27);
  a27=-5.3980611462160466e-01;
  a10=(a27*a25);
  a14=(a14+a10);
  a10=(a15-a16);
  a14=(a14*a10);
  a2=(a2+a14);
  a18=(a18*a5);
  a28=(a28*a9);
  a18=(a18+a28);
  a6=(a6*a13);
  a18=(a18+a6);
  a26=(a26*a17);
  a18=(a18+a26);
  a26=3.3732457471935139e-07;
  a26=(a26*a21);
  a18=(a18+a26);
  a26=-2.4075004404592054e-08;
  a6=(a26*a25);
  a18=(a18+a6);
  a6=(a19-a20);
  a18=(a18*a6);
  a2=(a2+a18);
  a22=(a22*a5);
  a29=(a29*a9);
  a22=(a22+a29);
  a30=(a30*a13);
  a22=(a22+a30);
  a27=(a27*a17);
  a22=(a22+a27);
  a26=(a26*a21);
  a22=(a22+a26);
  a26=4.6706108826693604e-01;
  a26=(a26*a25);
  a22=(a22+a26);
  a26=(a23-a24);
  a22=(a22*a26);
  a2=(a2+a22);
  a1=(a1-a2);
  if (res[0]!=0) res[0][0]=a1;
  if (res[0]!=0) res[0][1]=a0;
  a3=(a4-a3);
  if (res[0]!=0) res[0][2]=a3;
  a7=(a8-a7);
  if (res[0]!=0) res[0][3]=a7;
  a11=(a12-a11);
  if (res[0]!=0) res[0][4]=a11;
  a15=(a16-a15);
  if (res[0]!=0) res[0][5]=a15;
  a19=(a20-a19);
  if (res[0]!=0) res[0][6]=a19;
  a23=(a24-a23);
  if (res[0]!=0) res[0][7]=a23;
  a23=5.4184786818321906e-01;
  a19=(a23*a0);
  a19=(a4+a19);
  if (res[0]!=0) res[0][8]=a19;
  a19=2.2732385197231011e-01;
  a15=(a19*a0);
  a15=(a8+a15);
  if (res[0]!=0) res[0][9]=a15;
  a23=(a23*a0);
  a23=(a23-a4);
  if (res[0]!=0) res[0][10]=a23;
  a19=(a19*a0);
  a19=(a19-a8);
  if (res[0]!=0) res[0][11]=a19;
  a19=2.0481784543190165e-01;
  a23=(a19*a0);
  a23=(a12+a23);
  if (res[0]!=0) res[0][12]=a23;
  a23=8.0988154313854899e-01;
  a15=(a23*a0);
  a15=(a16+a15);
  if (res[0]!=0) res[0][13]=a15;
  a15=3.9463455784000195e-01;
  a11=(a15*a0);
  a11=(a20+a11);
  if (res[0]!=0) res[0][14]=a11;
  a11=1.5742998630435918e+00;
  a7=(a11*a0);
  a7=(a24+a7);
  if (res[0]!=0) res[0][15]=a7;
  a19=(a19*a0);
  a19=(a19-a12);
  if (res[0]!=0) res[0][16]=a19;
  a23=(a23*a0);
  a23=(a23-a16);
  if (res[0]!=0) res[0][17]=a23;
  a15=(a15*a0);
  a15=(a15-a20);
  if (res[0]!=0) res[0][18]=a15;
  a11=(a11*a0);
  a11=(a11-a24);
  if (res[0]!=0) res[0][19]=a11;
  a11=arg[1]? arg[1][0] : 0;
  a24=7.6694561124289573e-01;
  a15=(a24*a0);
  a15=(a11+a15);
  if (res[0]!=0) res[0][20]=a15;
  a15=arg[1]? arg[1][1] : 0;
  a20=7.6693189676533913e-01;
  a23=(a20*a0);
  a23=(a15+a23);
  if (res[0]!=0) res[0][21]=a23;
  a24=(a24*a0);
  a24=(a24-a11);
  if (res[0]!=0) res[0][22]=a24;
  a20=(a20*a0);
  a20=(a20-a15);
  if (res[0]!=0) res[0][23]=a20;
  a20=arg[1]? arg[1][2] : 0;
  if (res[0]!=0) res[0][24]=a20;
  a20=arg[1]? arg[1][3] : 0;
  if (res[0]!=0) res[0][25]=a20;
  a20=arg[1]? arg[1][4] : 0;
  if (res[0]!=0) res[0][26]=a20;
  a20=arg[1]? arg[1][5] : 0;
  if (res[0]!=0) res[0][27]=a20;
  a20=arg[1]? arg[1][6] : 0;
  if (res[0]!=0) res[0][28]=a20;
  a20=arg[1]? arg[1][7] : 0;
  if (res[0]!=0) res[0][29]=a20;
  a20=arg[1]? arg[1][8] : 0;
  if (res[0]!=0) res[0][30]=a20;
  a20=arg[1]? arg[1][9] : 0;
  if (res[0]!=0) res[0][31]=a20;
  a20=1.6000000000000000e-01;
  a15=1.;
  a15=(a4-a15);
  a15=casadi_sq(a15);
  a24=1.2500000000000000e+00;
  a11=(a8-a24);
  a11=casadi_sq(a11);
  a15=(a15+a11);
  a15=sqrt(a15);
  a15=(a20-a15);
  a11=5.4242866255517297e-01;
  a11=(a11*a0);
  a15=(a15+a11);
  if (res[0]!=0) res[0][32]=a15;
  a15=1.5000000000000000e+00;
  a15=(a4-a15);
  a15=casadi_sq(a15);
  a24=(a8-a24);
  a24=casadi_sq(a24);
  a15=(a15+a24);
  a15=sqrt(a15);
  a15=(a20-a15);
  a24=5.4243336345060678e-01;
  a24=(a24*a0);
  a15=(a15+a24);
  if (res[0]!=0) res[0][33]=a15;
  a15=1.2000000000000000e+00;
  a4=(a4-a15);
  a4=casadi_sq(a4);
  a15=1.7500000000000000e+00;
  a8=(a8-a15);
  a8=casadi_sq(a8);
  a4=(a4+a8);
  a4=sqrt(a4);
  a20=(a20-a4);
  a4=5.4222437246644040e-01;
  a4=(a4*a0);
  a20=(a20+a4);
  if (res[0]!=0) res[0][34]=a20;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v2_constr_h_0_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v2_constr_h_0_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v2_constr_h_0_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v2_constr_h_0_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v2_constr_h_0_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v2_constr_h_0_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v2_constr_h_0_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_RAMPC_v2_constr_h_0_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v2_constr_h_0_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_RAMPC_v2_constr_h_0_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_RAMPC_v2_constr_h_0_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v2_constr_h_0_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_RAMPC_v2_constr_h_0_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v2_constr_h_0_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_RAMPC_v2_constr_h_0_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_RAMPC_v2_constr_h_0_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
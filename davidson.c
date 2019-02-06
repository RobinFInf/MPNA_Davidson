#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>
#include <stdint.h>
#include <lapacke.h>

struct vecteur {
  int size;
  double* T;
};
typedef struct vecteur vecteur;

struct matrice{
  int ligne;
  int colonne;
  double** M;
};
typedef struct matrice matrice;

vecteur init_vecteur(int a, double val)
{
  vecteur x;
  int i;
  x.size = a;
  x.T = malloc(a*sizeof(double));
  for(i=0.0;i<x.size;i++)
  {
    x.T[i]=val;
  }
  return x;
}

matrice init_matrice(int a, int b, double val)
{
  matrice M;
  int i,j;
  M.ligne = a;
  M.colonne = b;
  M.M = malloc(b*sizeof(double*));
  for(i=0;i < b;i++)
  {
    M.M[i]=malloc(a*sizeof(double));
  }
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < M.colonne; j++)
    {
	     if (i == j)
       {
               M.M[i][j] = val+val*i;
       }else{
               M.M[i][j] = 0;
       }
    }
  }
  return M;
}
matrice init_matrice_ident(int a, int b)
{
  matrice M;
  int i,j;
  M.ligne = a;
  M.colonne = b;
  M.M = malloc(b*sizeof(double*));
  for(i=0;i < b;i++)
  {
    M.M[i]=malloc(a*sizeof(double));
  }
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < M.colonne; j++)
    {
	     if (i == j)
       {
               M.M[i][j] = 1;
       }else{
               M.M[i][j] = 0;
       }
    }
  }
  return M;
}
vecteur prod_matrice_vecteur(matrice m, vecteur v)
{
  vecteur res;
  int i, j;
  res = init_vecteur(v.size, 0.0);
  for (i=0;i<m.ligne;i++)
  {
    for (j=0;j<m.colonne;j++)
    {
      res.T[i] += m.M[i][j] * v.T[j];
    }
  }
  return res;
}

//Complexité temporelle 3n (parcours de 3 tableaux differents)
//Complexité spatiale 3n (3 tableaux de meme taille)
vecteur add(vecteur a, vecteur b)
{
  int i;
  if(a.size != b.size)
  {
    printf("Erreur vecteur de taille differentes\n");
    exit(0);
  }
  vecteur tmp = init_vecteur(a.size,0.0);
  for(i = 0; i < a.size;i++)
  {
    tmp.T[i]=a.T[i]+b.T[i];
  }
  return tmp;
}

//Complexité temporelle 2n (parcours de 2 tableaux différents)
//Complexité spatiale 2n (2 tableaux de meme taille)
vecteur scal_vect(double d, vecteur a)
{
  int i;
  vecteur tmp = init_vecteur(a.size,0.0);
  for(i = 0; i < a.size; i++)
  {
    tmp.T[i]=d*a.T[i];
  }
  return tmp;
}

double prod_scal(vecteur a, vecteur b)
{
  double res;
  int i;
  if(a.size != b.size)
  {
    printf("Erreur vecteur de taille differentes\n");
    exit(0);
  }
  for(i = 0; i < a.size; i++)
  {
    res += a.T[i] * b.T[i];
  }
  return res;
}

double prod_scal_par(vecteur a, vecteur b)
{
  double res;
  int i;
  res = 0.0;
  if(a.size != b.size)
  {
    printf("Erreur vecteur de taille differentes\n");
    exit(0);
  }
  #pragma omp parallel for reduction(+:res)
  for(i = 0; i < a.size; i++) {
    res += a.T[i]*b.T[i];
    }
    return res;
  }

double prod_scal_par_vect_mat(vecteur a, matrice m, int rang)
{
  double res;
  int i;
  res = 0.0;
  if(a.size != m.colonne)
  {
    printf("Erreur vecteur de taille differentes\n");
    exit(0);
  }
  #pragma omp parallel for reduction(+:res)
  for(i = 0; i < a.size; i++) {
    res += a.T[i]*m.M[rang][i];
    }
    return res;
}
double prod_scal_par_mat_mat(matrice a, matrice m, int rangi, int rangj)
{
  double res;
  int i;
  res = 0.0;
  if(a.ligne != m.colonne)
  {
    printf("Erreur vecteur de taille differentes\n");
    exit(0);
  }
  #pragma omp parallel for reduction(+:res)
  for(i = 0; i < a.ligne; i++) {
    res += a.M[rangi][i]*m.M[i][rangj];
    }
    return res;
}
vecteur prod_matrice_vecteur_par(matrice m, vecteur v )
{
  vecteur res;
  int i;
  res = init_vecteur(v.size, 0.0);
  for (i=0;i<m.ligne;i++)
  {
    res.T[i] = prod_scal_par_vect_mat(v,m,i);
  }
  return res;
}
matrice prod_matrice_matrice_par(matrice m, matrice p )
{
  matrice res;
  int i, j;
  res = init_matrice(m.ligne, p.colonne, 0.0);
  for (i=0;i<m.ligne;i++)
  {
    for (j=0; j<m.colonne; j++)
    {
      res.M[i][j] = prod_scal_par_mat_mat(m,p,i,j);
    }
  }
  return res;
}
matrice prod_matrice_matrice(matrice m, matrice p )
{
  matrice res;
  int i, j, k;
  res = init_matrice(m.ligne, p.colonne, 0.0);
  for (i=0;i<m.ligne;i++)
  {
    for (j=0; j<m.colonne; j++)
    {
      for (k=0; k < m.ligne; k++)
      {
        res.M[i][j] += m.M[i][k]*p.M[k][j];
      }
    }
  }
  return res;
}
matrice col(vecteur v)
{
  matrice M;
  int i,j;
  M.ligne = v.size;
  M.colonne = 1;
  M.M = malloc(sizeof(double*));
  for(i=0; i<1; i++)
  {
    M.M[i]=malloc(v.size*sizeof(double));
  }
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < 1 ; j++)
    {
               M.M[i][j] = v.T[i];
    }
  }
  return M;
}
vecteur soustraction(vecteur v1, vecteur v2)
{
  vecteur res;
  int i;
  res = init_vecteur(v1.size, 0.0);
  for (i=0; i<v1.size; i++)
  {
    res.T[i] = v1.T[i] - v2.T[i];
  }
  return res;
}

void print_matrix(char* desc, int m, int n, double* a, int lda){
  int i, j;
  printf("%s\n", desc);
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      printf("%6.2f\n", a[i+j*lda]);
    }
  }
}

void davidson(int N)
{
    int j,k;
    matrice A, H, Da, i, tmpY;
    vecteur v[N];
    A = init_matrice(N,N,2.0);
    Da = init_matrice(N,N,2.0);
    i = init_matrice_ident(N,N);
    H = init_matrice(N,N,0.0);
    v[0] = init_vecteur(N,1.0);
    vecteur w[N];
    vecteur r,y;
    double theta;
    double resVal[N];
    double resVect[N];
    double uH[5*5] = {
      0.67, -0.20, 0.19, -1.06, 0.46,
      0.0, 3.82, -0.13, 1.06, -0.48,
      0.0, 0.0, 3.27, 0.11, 1.10,
      0.0, 0.0, 0.0, 5.86, -0.98,
      0.0, 0.0, 0.0, 0.0, 3.54
    };
    int isuppz[2*N];
    int info;
    int m[1];
    m[0] = 0;
    for(j=0; j < N ; j++)
    {
      w[j]=prod_matrice_vecteur(A,v[j]);
      for(k = 0; k<j; k++)
      {
        H.M[k][j] = prod_scal(v[k],w[j]);
        H.M[j][k] = prod_scal(v[j],w[k]);
      }
      H.M[j][j] = prod_scal(v[j],w[j]);

      //CALCUL DES EIGENVALUE ET DU VECTEUR theta et s//

      //H triangulaire sup dans uH//
      /*
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          if (j >= i) {
            uH[i*j] = H.M[i][j];
          }else{
            uH[i*j] = 0.0;
          }
        }
      }*/
      /////////////////////////////
      info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', N*N, uH, N, 0.0, 0.0, 1, N, -1.0, m, resVal, resVect, N, isuppz);
      //return 0 = succes, -# wrong parameter, others error.
      printf("dsyevr statut : %d\n", info);
      printf("nbr Eigen value : %2i\n", m);
      print_matrix("Eigen value", 1, N, resVal, 1);
      print_matrix("Eigen vector", N, N, resVect, N);
      //////////////////////////////////////////////////

      /*
      tmpY = col(v[j]);
      y = prod_matrice_vecteur(tmpY, s);
      r = soustraction(prod_matrice_vecteur(A,y),scal_vect(theta,i));
      t =
      */
      // inversion de matrice LAPACKE_dgetrf et LAPACKE_dgetri

    }
}

int main(int argc, char const *argv[]) {
  uint64_t  useconde_start, useconde_stop, time_elapsed;
  struct timeval tv;
  time_elapsed = 0;
  int i,N; // N la taille de la matice A(N*N)
  N = 5;
  for (i = 0; i<10; i++)
  {
    gettimeofday(&tv, NULL);
    useconde_start = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    davidson(N);
    gettimeofday(&tv, NULL);
    useconde_stop = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    time_elapsed += (useconde_stop - useconde_start);
  }
  printf("Time Elapsed : %ld ms \n",time_elapsed/10);
  return 0;
}

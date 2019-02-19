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
#include <math.h>
#include <mpi.h>

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
  M.M = malloc(a*sizeof(double*));
  for(i=0;i < a;i++)
  {
    M.M[i]=malloc(b*sizeof(double));
  }
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < M.colonne; j++)
    {
	     if (i == j)
       {
               M.M[i][j] = val;
       }else{
               M.M[i][j] = 0;
       }
    }
  }
  return M;
}

matrice init_matrice_test(int a, int b)
{
  matrice M;
  int i,j;
  M.ligne = a;
  M.colonne = b;
  M.M = malloc(a*sizeof(double*));
  for(i=0;i < a;i++)
  {
    M.M[i]=malloc(b*sizeof(double));
  }
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < M.colonne; j++)
    {
	     if (i == j)
       {
               M.M[i][j] = i+1;
       }else{
               M.M[i][j] = 0;
       }
    }
  }
  for(i=0; i<M.ligne; i++)
  {
    if( i != M.ligne -1)
    {
      M.M[i][i+1] = -0.1;
    }
    if(i != 0)
    {
      M.M[i][i-1] = 0.1;
    }
  }
  return M;
}


int pmap(int i, int size, matrice m){
  size = size-1;
  int r = (int)ceil((double)m.ligne / (double)size);
  int proc = i/r;
  return proc+1;
}

void print_matrice(matrice M)
{
  for(int i = 0; i<M.ligne; i++)
  {
    for (int j = 0; j<M.colonne; j++)
    {
      printf("%lf ",M.M[i][j]);
    }
    printf("\n");
  }
}
void print_vecteur(vecteur v)
{
  for(int i=0; i<v.size; i++)
  {
    printf ("%lf ",v.T[i]);
  }
  printf("\n");
}

vecteur prod_matrice_vecteur_MPI(matrice m, vecteur v)
{
  int size, rank;
  MPI_Status stat;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  vecteur res;
  res = init_vecteur(v.size, 0.0);

  if (rank == 0) {
    for (int i = 0; i < m.ligne; i++) {
      int proc = pmap(i, size, m);
      MPI_Send(m.M[i], m.colonne, MPI_DOUBLE, proc, (100*(i+1)), MPI_COMM_WORLD);
    }
    for (int i = 0; i < m.ligne; i++) {
      int sender_proc = pmap(i, size, m);
      MPI_Recv(&res.T[i], m.colonne, MPI_DOUBLE, sender_proc, i, MPI_COMM_WORLD, &stat);
    }
  }else{
    for (int i = 0; i < m.ligne; i++) {
      int proc = pmap(i, size, m);
      if (rank == proc) {
        double b[m.colonne];
        MPI_Recv(b, m.colonne, MPI_DOUBLE, 0, (100*(i+1)), MPI_COMM_WORLD, &stat);
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < m.colonne; j++) {
          sum = sum + (b[j] * v.T[j]);
        }
        MPI_Send(&sum, 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
      }
    }
  }
  return res;
}

matrice prod_matrice_matrice_MPI(matrice m, matrice m2)
{
  int size, rank;
  MPI_Status stat;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matrice res;
  res = init_matrice(m.ligne, m2.colonne, 0.0);

  if (rank == 0) {
    for (int i = 0; i < m.ligne; i++) {
      int proc = pmap(i, size, m);
      MPI_Send(m.M[i], m.colonne, MPI_DOUBLE, proc, (100*(i+1)), MPI_COMM_WORLD);
    }
    for (int i = 0; i < m.ligne; i++) {
      int sender_proc = pmap(i, size, m);
      MPI_Recv(res.M[i], m2.colonne, MPI_DOUBLE, sender_proc, i, MPI_COMM_WORLD, &stat);
    }
  }else{
    for (int i = 0; i < m.ligne; i++) {
      int proc = pmap(i, size, m);
      double c[m2.colonne];
      if (rank == proc) {
        double b[m.colonne];
        MPI_Recv(b, m.colonne, MPI_DOUBLE, 0, (100*(i+1)), MPI_COMM_WORLD, &stat);
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < m.colonne; j++) {
          double sum = 0.0;
          for (int k = 0; k < m.colonne; k++) {
            sum = sum + (b[k] * m2.M[k][j]);
          }
          c[j] = sum;
        }
        MPI_Send(c, m2.colonne, MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
      }
    }
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

int main(int argc, char *argv[]) {
  uint64_t  useconde_start, useconde_stop, time_elapsed;
  struct timeval tv;
  time_elapsed = 0;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int N = 5;
  matrice A, res;
  vecteur v;//, res;
  A = init_matrice_test(N,N);
  v = init_vecteur(N,2.0);
  if (rank == 0) {
    print_matrice(A);
    printf("LOL\n");
  }
  for (int i = 0; i<10; i++)
  {
    gettimeofday(&tv, NULL);
    useconde_start = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    //res = prod_matrice_vecteur_MPI(A, v);
    //prod_matrice_vecteur_par(A, v);
    res = prod_matrice_matrice_MPI(A, A);
    gettimeofday(&tv, NULL);
    useconde_stop = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    time_elapsed += (useconde_stop - useconde_start);
  }
  if (rank == 0) {
    printf("Time Elapsed : %ld ms \n",time_elapsed/10);
    print_matrice(res);
  }

  MPI_Finalize();
  return 0;
}

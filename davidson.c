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
matrice scal_mat(double d, matrice a)
{
  int i,j;
  matrice tmp = init_matrice(a.ligne,a.colonne,0.0);
  for (i=0;i < a.ligne; i++)
  {
    for(j=0;j < a.colonne ; j++)
    {
               tmp.M[i][j] = d*a.M[i][j];
    }
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
vecteur sous_vect(vecteur v1, vecteur v2)
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

matrice sous_mat(matrice a, matrice b)
{
  matrice M;
  int i,j;
  M = init_matrice(a.ligne, a.colonne, 0.0);
  for (i=0;i < M.ligne; i++)
  {
    for(j=0;j < M.colonne ; j++)
    {
               M.M[i][j] = a.M[i][j] - b.M[i][j];
    }
  }
  return M;
}
void d_v(double * d, vecteur v)
{
  int i;
  for(i=0; i<v.size; i++)
  {
    v.T[i]=d[i];
  }
}
void m_d(matrice a, double* d)
{
  int i,j;
  for(i=0; i<a.ligne; i++)
  {
    for(j=0; j<a.ligne; j++)
    {
      ;
    }
  }
}

void init_lapack(int n, double* a, matrice m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i*n+j] = m.M[i][j];
    }
  }
}
void davidson(int N)
{
    int j,k;
    matrice A, H, Da, id, tmpY, inverse;
    vecteur v[N];
    A = init_matrice(N,N,2.0);
    Da = init_matrice(N,N,2.0);
    id = init_matrice_ident(N,N);
    H = init_matrice(N,N,0.0);
    v[0] = init_vecteur(N,1.0);
    vecteur w[N];
    vecteur r,y,s;
    s = init_vecteur(N,0.0);
    double theta;
    //////////////////////////////////////////////////
    int n = N, lda = N, ldvl = N, ldvr = N, info, maximum;
    double wr[n], wi[n], vl[ldvl*n], vr[ldvr*n];
    double *a = malloc(N*N*sizeof(double));

    /*
    double a[5*5] = {
           -1.01,  0.86, -4.60,  3.31, -4.81,
            3.98,  0.53, -7.04,  5.29,  3.55,
            3.30,  8.26, -3.89,  8.20, -1.51,
            4.43,  4.96, -7.66, -7.33,  6.18,
            7.31, -6.43, -6.16,  2.47,  5.58
    };*/
    //////////////////////////////////////////////////
    for(j=0; j < N ; j++)
    {
      w[j]=prod_matrice_vecteur(A,v[j]);
      for(k = 0; k<j; k++)
      {
        H.M[k][j] = prod_scal(v[k],w[j]);
        H.M[j][k] = prod_scal(v[j],w[k]);
      }
      H.M[j][j] = prod_scal(v[j],w[j]);
      /* Solve eigenproblem */
      init_lapack(N,a,A);
      info = LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, wr, wi,
                          vl, ldvl, vr, ldvr );
      /* vérifie la convergence */
      if( info > 0 ) {
              printf( "Failed to compute eigenvalues.\n" );
              exit( 1 );
      }
      maximum = max(wr, n);
      print_eigenvalues( "Eigenvalues", n, wr, wi );
      //print_eigenvectors( "Left eigenvectors", n, wi, vl, ldvl );
      print_eigenvectors( "Right eigenvectors", n, wi, vr, ldvr );

      printf("Max EigenValue : %2f\n", wr[maximum]);
      printf("EigenVector : %2f\n", vr[maximum]);
      theta =  wr[maximum]; // recupere la valeur propre max , les autres sont dans wr
      d_v(vr,s); // Remet le vecteur dans la structure de donnée
      tmpY = col(v[j]); // Transforme un vecteur en matrice
      y = prod_matrice_vecteur(tmpY, s);
      r = sous_vect(prod_matrice_vecteur(A,y),scal_vect(theta,y));
      inverse = sous_mat(Da,scal_mat(theta,id));
      // inversion de matrice LAPACKE_dgetrf et LAPACKE_dgetri
      int pivotArray[N+1];
      lapack_int err;
      double inverse_lapack[N*N];
      init_lapack(N,inverse_lapack,inverse);
      err = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,inverse_lapack,N,pivotArray);
      if (err !=0)
      {
        printf("erreur sur LAPACKE_dgetrf \n");
        exit( 1 );
      }
      err = LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,inverse_lapack,N,pivotArray);
      printf( "inversion faite;");

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

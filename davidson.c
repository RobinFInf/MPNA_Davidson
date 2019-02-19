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
//#include <mpi.h>
#define P  0.000001

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
               M.M[i][j] = i%10;
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
  M = init_matrice(v.size,1,0.0);
  for (i=0; i< M.ligne; i++)
  {
    for(j=0;j < 1 ; j++)
    {
               M.M[i][j] = v.T[i];
    }
  }
  return M;
}
void print_eigenvalues( char* desc, int n, double* wr, double* wi ) {
        int j;
        printf( "\n %s\n", desc );
   for( j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         printf( " %2f", wr[j] );
      } else {
         printf( " (%2f,%2f)", wr[j], wi[j] );
      }
   }
   printf( "\n" );
}

void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv ) {
        int i, j;
        printf( "\n %s\n", desc );
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            printf( " %2f", v[i*ldv+j] );
            j++;
         } else {
            printf( " (%2f,%2f)", v[i*ldv+j], v[i*ldv+(j+1)] );
            printf( " (%2f,%2f)", v[i*ldv+j], -v[i*ldv+(j+1)] );
            j += 2;
         }
      }
      printf( "\n" );
   }
}

int max(double *tab, int n)
{
  double max = -9999999.999;
  int tmp;
  for (int i = 0; i < n; i++) {
    if (tab[i] > max) {
      max = tab[i];
      tmp = i;
    }
  }
  return tmp;
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
vecteur norm(vecteur v)
{
  vecteur res;
  int i;
  double tot;
  res = init_vecteur(v.size, 0.0);
  for(i=0; i<v.size; i++)
  {
    tot += v.T[i]*v.T[i];
  }
  tot = sqrt(fabs(tot));
  for(i=0; i<v.size; i++)
  {
  res.T[i]= v.T[i]/tot;
  }
  return res;
}

void init_lapack(int n, double* a, matrice m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[i*n+j] = m.M[i][j];
    }
  }
}

void return_matrice(int n, double * d, matrice m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
     m.M[i][j] = d[i*n+j];
    }
  }
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
void re_init(double * v, int N)
{
  for (int i=0 ; i<N; i++)
  {
    v[i] = 0.0;
  }
}

/*
int pmap(int i, int size, matrice m){
  size = size-1;
  int r = (int)ceil((double)m.ligne / (double)size);
  int proc = i/r;
  return proc+1;
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

*/

void davidson(matrice A,int N, double wr[N], double vr[N*N], int nb_eig)
{
    int j,k;
    matrice  H, Da, id, tmpY, inverse;
    vecteur v[N];
    Da = init_matrice(N,N,2.0);
    id = init_matrice_ident(N,N);
    H = init_matrice(N,N,0.0);
    v[0] = init_vecteur(N,1.0);
    vecteur w[N];
    vecteur r, y, s, t, tmpT;
    s = init_vecteur(N,0.0);
    double theta, convergence;
    int n = N, lda = N, ldvl = N, ldvr = N, info;
    int maximum;
    double wi[n], vl[ldvl*n];
    double a[N*N];
    //print_matrice(A);
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
      init_lapack(N,a,H);;
      info = LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, wr, wi,
                          vl, ldvl, vr, ldvr );

      /* vérifie la convergence */
      if( info > 0 ) {
              printf( "Failed to compute eigenvalues.\n" );
              exit( 1 );
      }
      maximum = max(wr, n);
      theta =  wr[maximum]; // recupere la valeur propre max , les autres sont dans wr
      if ( j == 0)
      {
        convergence = theta;
      }else if (theta > convergence)
      {
        if (P > theta - convergence)
        {
          printf("convergence ! \n");
          break;
        }else{
          convergence = theta;
        }
      }else if(theta < convergence)
      {

      if (P >  convergence - theta)
      {
        printf("convergence ! \n");
        break;
      }else{
        convergence = theta;
      }
    }else{
      printf("convergence ! \n");
      break;
    }

      d_v(vr,s); // Remet le vecteur dans la structure de donnée
      tmpY = col(v[j]); // Transforme un vecteur en matrice
      y = prod_matrice_vecteur(tmpY, s);
      r = sous_vect(prod_matrice_vecteur(A,y),scal_vect(theta,y));
      //print_vecteur(r);
      inverse = sous_mat(Da,scal_mat(theta,id));
      // inversion de matrice LAPACKE_dgetrf et LAPACKE_dgetri
      int pivotArray[N+1];
      lapack_int err;
      double inverse_lapack[N*N];
      init_lapack(N,inverse_lapack,inverse);
      err = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,inverse_lapack,N,pivotArray);
      if (err !=0 )
      {
        printf("erreur sur LAPACKE_dgetrf err=%d \n",err);
        //exit( 1 );
      }

      err = LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,inverse_lapack,N,pivotArray);
      //printf( "inversion faite \n");
      return_matrice(N, inverse_lapack, inverse);
      t = prod_matrice_vecteur(inverse,r);
      double uu;
      uu = prod_scal(v[j],v[j]);
      tmpT = scal_vect(uu,t);
      t = sous_vect(t,tmpT);
      //print_vecteur(t);
      v[j+1] = norm(t);
      printf("itération  numeros %d \n",j);
    }
    free(v);
}
void test(matrice A, int N, int nb_eig)
{
  int n = N, lda = N, ldvl = N, ldvr = N, info;
  int maximum;
  double wi[n], vl[ldvl*n];
  double a[N*N];
  double test_eigen[N];
  double test_vect[N*N];
  init_lapack(N,a,A);
  info = LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, a, lda, test_eigen, wi,
                      vl, ldvl, test_vect, ldvr );
  /* vérifie la convergence */
  if( info > 0 ) {
          printf( "Failed to compute eigenvalues.\n" );
          exit( 1 );
  }
  maximum = max(test_eigen, n);
  printf("EigenValue Test:");
  for(int l=0; l<nb_eig; l++)
  {
    printf(" %2f \n", test_eigen[maximum+l]);
  }
  printf ("\n");
}
int main(int argc, char const *argv[]) {
  uint64_t  useconde_start, useconde_stop, time_elapsed;
  struct timeval tv;
  time_elapsed = 0;
  int N, nb_eig; // N la taille de la matice A(N*N)
  N =399;
  nb_eig = 5;
  int maximum;
  matrice A; // initialisation de la matrice
  A = init_matrice_test(N,N);
  double wr[N],vr[N*N];
//  for (int i = 0; i<10; i++)
//  {
    gettimeofday(&tv, NULL);
    useconde_start = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    davidson(A,N,wr,vr,nb_eig);
    gettimeofday(&tv, NULL);
    useconde_stop = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
    time_elapsed += (useconde_stop - useconde_start);
//  }
  printf("Time Elapsed : %ld ms \n",time_elapsed/10);
  test(A,N,nb_eig);
  maximum = max(wr, N);
  printf("EigenValue Davidson:");
  for(int l=0; l<nb_eig; l++)
  {
    printf(" %2f \n", wr[maximum+l]);
  }
  printf ("\n");
  return 0;
}

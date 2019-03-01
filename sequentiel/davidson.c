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
#define P  0.0001


int convergence;
double * wr;

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
  int i,j,alea;
  srand(time(NULL)); // initialisation de rand
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
       alea = rand()%(a/2);
	     if (i == j)
       {
               M.M[i][j] = i%5;
       }else{
            if(alea < 0.20 * (a/2))
            {
                 M.M[i][j] = rand()%10;
            }
            else{
                 M.M[i][j] = 0.0;
            }
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

matrice init_matrice_test_precision(int a, int b)
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
void d_v(double * d, vecteur v, int maximum)
{
  int i;
  for(i=0; i<v.size; i++)
  {
    v.T[i]=d[i+v.size*maximum];
  }
}
double norm(vecteur v)
{
  int i;
  double tot;
  tot = v.T[0]*v.T[0];
  for(i=1; i<v.size; i++)
  {
    tot += v.T[i]*v.T[i];
  }
  tot = sqrt(fabs(tot));
  return tot;
}

vecteur norm2(vecteur v)
{
  int i;
  double tot;
  vecteur res;
  res = init_vecteur(v.size,0.0);
  tot = v.T[0]*v.T[0];
  for(i=1; i<v.size; i++)
  {
    tot += v.T[i]*v.T[i];
  }
  tot = sqrt(fabs(tot));
  for(i = 0; i<v.size; i++)
  {
    res.T[i] = v.T[i]/tot;
  }
  return res;
}
void init_lapack2(int n, double* a, matrice m)
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      a[(i*n)+j] = m.M[i][j];
    }
  }
}

void init_lapack(int n, double* a, matrice m , matrice A)
{
  for (int i = 0; i < A.ligne; i++) {
    for (int j = 0; j < A.colonne; j++) {
      a[(i*n)+j] = m.M[i][j];
    }
  }
}
void return_matrice(int n, double * d, matrice m)
{
  for (int i = 0; i < m.ligne; i++) {
    for (int j = 0; j < m.colonne; j++) {
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
vecteur projection(vecteur v, vecteur u) // Algo wikipédia de gram-schmitz
{
  double tmp;
  tmp = prod_scal(u,v);
  tmp = tmp / (prod_scal(u,u));
  vecteur res;
  res = scal_vect(tmp,u);
  return res;
}
vecteur orthonormalize(vecteur* a, vecteur b, int j)
{
  if (j == 0)
  {
    //printf("\n 1 \n");
    return b;
  }
  if ( j == 1)
  {
    vecteur res = init_vecteur(b.size, 0.0);
    res = sous_vect(b,projection(b,a[0]));
    //printf("\n 2 \n");
    return res;
    }else{
    vecteur res = init_vecteur(b.size, 0.0);
    res = sous_vect(b,projection(b,a[0]));
    for ( int i = 1; i < j; i++ )
    {
    res = sous_vect(res,projection(b,a[i]));
    }
    //printf("\n 3 \n");
    return res;
  }
}
void ajout_col(vecteur v, int n, matrice a)
{
  int i;
  for(i=0; i<v.size; i++)
  {
    a.M[i][n] = v.T[i];
  }
}
matrice transpose(matrice a)
{
  int i,j;
  matrice b;
  b=init_matrice(a.ligne,a.colonne,0.0);
  for (i=0; i<a.ligne; i++)
  {
    for(j=0; j<a.colonne; j++)
    {
      b.M[i][j] = a.M[j][i];
    }
  }
  return b;
}
vecteur davidson(matrice A,int N, double wr[N], double vr[N*N], int nb_eig, vecteur ritz)
{
    //INITIALISATION////////////////////////
    int j, n = N, lda = N, ldvl = N, ldvr = N, info, maximum, pivotArray[N], err;
    double theta, wi[n], vl[ldvl*n], h[N*N], inverse_lapack[N*N];
    matrice  H, Da, id, inverse, W, V;
    vecteur r, s, t, v[N];
    Da = init_matrice(N,N,2.0);
    id = init_matrice_ident(N,N);
    H = init_matrice(N,N,0.0);
    W = init_matrice(N,N,0.0);
    V = init_matrice(N,N,0.0);
    v[0] = ritz;
    s = init_vecteur(N,0.0);
    ////////////////////////////////////////////////////////////////////////////////////:
    for(j=0; j < N-1 ; j++)
    {
      ajout_col(v[j], j, V);
      W=prod_matrice_matrice(A,V);
      H=prod_matrice_matrice(transpose(V),W);
      init_lapack(N,h,A,H);
      // on cherche les valeurs propres de la matrice H
      info = LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'V', 'V', n, h, lda, wr, wi,
                          vl, ldvl, vr, ldvr );
      if( info > 0 ) {
              printf( "Failed to compute eigenvalues.\n" );
              exit( 1 );
      }
      maximum = max(wr, n);
      theta =  wr[maximum]; // recupere la valeur propre max , les autres sont dans wr
      d_v(vr,s,maximum); // Remet le vecteur dans la structure de donnée
      ritz = prod_matrice_vecteur(V,s);
      r = sous_vect(prod_matrice_vecteur(A,ritz),scal_vect(theta,ritz));
      /////////////////Fin de la procedure de rayleiht-ritz ///////////////
      if (norm(r) < P && j == N/2) // On garde une certaine précison en vérifiant la convergence
      {
        convergence = 1;
        return ritz;
      }
      inverse = sous_mat(Da,scal_mat(theta,id));
      init_lapack2(N,inverse_lapack,inverse);
      // On calcul le vecteur de redémarage
      err = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,inverse_lapack,N,pivotArray);
      if (err !=0 )
      {
        printf("erreur sur LAPACKE_dgetrf err=%d \n",err);
        exit( 1 );
      }
      err = LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,inverse_lapack,N,pivotArray);
      return_matrice(N, inverse_lapack, inverse);
      t = prod_matrice_vecteur(inverse,r);
      v[j+1] =orthonormalize(v,t,j);
    }
    return ritz; // On redemare avec le vecteur de ritz
}

int main(int argc, char const *argv[]) {
  if (argc != 2)
  {
    printf("Ne pas oublier la taille de la matrice\n");
    exit(0);
  }
  uint64_t  useconde_start, useconde_stop, time_elapsed;
  struct timeval tv;
  time_elapsed = 0;
  int N, nb_eig; // N la taille de la matice A(N*N)
  N = atoi(argv[1]);
  printf("taille de Matrice = %d * %d \n",N,N);
  nb_eig = 1;
  int maximum;
  matrice A; // initialisation de la matrice
  A = init_matrice_test_precision(N,N);// fonction pour initialiser une matrice diagonale symétrique sinon remplir le champs M de la matrice
                                      //avec la fonction void return_matrice(int n, double * d, matrice m)
  double vr[N*N];
  wr = malloc(N*sizeof(double));
  vecteur ritz, tmp;
  ritz = init_vecteur(N,1.0);
  tmp = init_vecteur(N,0.0);
  convergence = 0;
  gettimeofday(&tv, NULL);
  useconde_start = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
  tmp = davidson(A,N,wr,vr,nb_eig,ritz);
  while (!convergence)
  {
    tmp = davidson(A,N,wr,vr,nb_eig,tmp);// on redemare jusqu'a la convergence
  }
  gettimeofday(&tv, NULL);
  useconde_stop = (tv.tv_sec * (uint64_t)1000) + (tv.tv_usec / 1000);
  time_elapsed += (useconde_stop - useconde_start);
  printf("Time Elapsed : %ld ms \n",time_elapsed);
  maximum = max(wr, N);
  printf("EigenValue Davidson:");
  for(int l=0; l<nb_eig; l++)
  {
    printf(" %2f \n", wr[maximum+l]);
  }
  printf ("\n");
  return 0;
}

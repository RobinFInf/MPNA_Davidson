# MPNA_Davidson

Dans chaque dossier vous pouvez utiliser la commande "make" pour compiler le programme.

Dans le dossier sequentiel utiliser la commande "./davidson.exe 50" pour exécuter le programme sur une matrice 50x50.

Dans le dossier OpenMP utiliser la commande "OMP_NUM_THREADS=4 ./davidson.exe 50" pour exécuter le programme avec une matrice de taille 50x50.

Dans le dossier MPI_OpenMP utiliser la commande "env OMP_NUM_THREADS=4 mpiexec -np 4 ./davidson.exe 50" pour exécuter le programme sur une matrice 50x50.

Il est possible de simplement utiliser la commande "make run" dans le dossier de la version voulu pour exécuter le programme après l'avoir compilé avec "make"
"make run" utilise par défaut des matrices de taille 50x50.

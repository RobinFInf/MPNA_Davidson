
davidson: davidson.o
	mpicc -fopenmp -o davidson.exe davidson.o -Wall -llapacke -lm -g
davidson.o: davidson_MPI.c
	mpicc -fopenmp -o davidson.o -c davidson_MPI.c -Wall -llapacke -lm -g
run :
	env OMP_NUM_THREADS=4 mpiexec -np 4 ./davidson.exe
clean :
	rm *.o *.exe


davidson: davidson.o
	gcc -fopenmp -o davidson.exe davidson.o -Wall -llapacke -lm -g
davidson.o: davidson_OpenMP.c
	gcc -fopenmp -o davidson.o -c davidson_OpenMP.c -Wall -llapacke -lm -g
run :
	OMP_NUM_THREADS=4 ./davidson.exe
clean :
	rm *.o *.exe

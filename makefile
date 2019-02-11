
davidson: davidson.o
	gcc -fopenmp -o davidson.exe davidson.o -Wall -llapacke -lm -g
davidson.o: davidson.c
	gcc -fopenmp -o davidson.o -c davidson.c -Wall -llapacke -lm -g
run :
	./davidson.exe
run2 :
	OMP_NUM_THREADS=2 ./davidson.exe
run4 :
	OMP_NUM_THREADS=4 ./davidson.exe
clean :
	rm *.o *.exe

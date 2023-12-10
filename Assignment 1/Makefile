build:
	g++ -fopenmp -O3 -Wall -Wextra -Wpedantic -std=c++20 iom.cpp main.cpp -o iom_cpp.out

debug:
	g++ -g -fopenmp iom.cpp main.cpp -o iom_cpp.out

clean:
	rm -f *.out *.gch

run:
	./iom_cpp.out ./test.in ./test.out 1

run2:
	./iom_cpp.out ./small_input.in ./small_input.out 1

perf:
	perf stat -- ./iom_cpp.out ./test.in ./test.out 8 > /dev/null

test:
	g++ -fopenmp -O3 -Wall -Wextra -Wpedantic -std=c++20 test.cpp -o test.o
	srun -p xs-4114 perf stat -r 1 -- ./test.o

seq:
	perf stat -- ./iom_sequential ./large_input.in ./large_input.out 8

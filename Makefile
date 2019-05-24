NVCC = nvcc

all: vecadd multiplicacion

%.o : %.cu
	$(NVCC) -c $< -o $@

vecadd : vecadd.o
	$(NVCC) $^ -o $@

multiplicacion: 
	$(NVCC) multiplicacion.cu -o multiplicacion.o

saxpyMatrix: 
	$(NVCC) saxpyMatrix.cu -o saxpyMatrix.o

clean:
	rm -rf *.o *.a vecadd

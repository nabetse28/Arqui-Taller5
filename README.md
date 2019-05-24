# Arqui-Taller5

## Usage
For use this programs you have to run somo of the commands that are in the Makefile, this Makefile was taken from [here](https://github.com/jefg89/helloCUDA) and was modified to use the following commands.

### multiplicacion.cu
This file does a multiplicaction of two matrixes of 4x4. The command to build the excecutable is:

```bash 
make multiplicacion
./multiplicacion.o
```
### saxpyMatrix.cu
This file does a saxpy algorithm with two matrixes. The commando to build the executable is:

```bash 
make saxpyMatrix
./saxpyMatrix.o
```

## Saxpy Matrix
### Matrix 4x4
![](images/saxpyMatrix4x4.png)

### Matrix 8x8
![](images/saxpyMatrix8x8.png)

### Matrix 16x16
![](images/saxpyMatrix16x16.png)

### Matrix 32x32
![](images/saxpyMatrix32x32.png)

### Statistics
The blue line is for the GPU and the red line is for the CPU.
![](images/statistics.png)

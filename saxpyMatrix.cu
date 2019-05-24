#include <stdio.h>
#include <cuda.h>

int *a, *b;  // host data
int *c, *c2;  // result


//Cuda error checking - non mandatory
void cudaCheckError() {
 cudaError_t e=cudaGetLastError();
 if(e!=cudaSuccess) {
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
   exit(0); 
 }
}

//GPU kernel 
__global__
void saxpyGPU(int *A, int *B, int *C, int N, int constelement){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    int element = 0;
    //Verify if it's in the bounds
    // if (row < N && col < N){
        //Dot product to compute and the element C[col, row]
        // for (int i = 0; i < N; ++i)
        // {
        element = (A[row*N + col] * constelement) + B[row*N + col];
        // }
    // }
    //Store result of the element
    C[row*N + col] = element;
}

//CPU function
void saxpyCPU(int *A, int *B, int *C, int N, int constelement){
    // int cont = 0;
    int element = 0;
    for (int i = 0; i < N ; ++i)
    {
        // printf("A element i: %d", A[i]);
        
        for (int k = 0; k < N; ++k)
        {
            // cont++;
            // printf("A element k: %d", A[k]);
            element = A[i*N + k] * constelement + B[i*N + k];
            C[i*N + k] = element;
        }
        
    }
    // printf("cont: %d\n\n", cont);
}

int main(int argc,char **argv)
{
    printf("Begin \n");
    //Iterations
    int n=16;

    int constelement = 43;
    //Number of blocks (4x4)
    int nBytes = n*n*sizeof(int);
    //Block size and number
    int block_size, block_no;

    //memory allocation 
    a = (int *) malloc(nBytes);
    b = (int *) malloc(nBytes);
    c = (int *) malloc(nBytes);
    c2 = (int *) malloc(nBytes);

    int *a_d,*b_d,*c_d;
    block_size = n; //threads per block
    block_no = n/block_size;
    
    //Work definition
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(block_no, block_no);

    // Data filling
    for(int i=0;i<n*n;i++)
    {
        a[i]= rand() % 100;
        b[i]= rand() % 100;
    }
        

    //Showing data
    // printf("Matrix A:\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     for (int j = 0; j < n; ++j)
    //     {
    //         printf("%d\t", a[i*n+j]);
    //     }
    //     printf("\n");
    // }
    // printf("\nMatrix B:\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     for (int j = 0; j < n; ++j)
    //     {
    //         printf("%d\t", b[i*n+j]);
    //     }
    //     printf("\n");
    // }

    printf("\n\nAllocating device memory on host..\n");
   //GPU memory allocation
    cudaMalloc((void **) &a_d, n*n*sizeof(int));
    cudaMalloc((void **) &b_d, n*n*sizeof(int));
    cudaMalloc((void **) &c_d, n*n*sizeof(int));

    printf("Copying to device..\n");
    cudaMemcpy(a_d, a, n*n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n*n*sizeof(int), cudaMemcpyHostToDevice);

    //Starting clock
    clock_t start_d=clock();
    printf("Doing GPU matrix multiplicationCPU\n\n");
    saxpyGPU<<<dimGrid,dimBlock>>>(a_d, b_d, c_d, n, constelement);
    cudaCheckError();

    //Wait for kernel call to finish
    cudaThreadSynchronize();

    clock_t end_d = clock();
    
    printf("Doing CPU Vector add\n");
    clock_t start_h = clock();
    saxpyCPU(a, b, c2, n, constelement);
    clock_t end_h = clock();


    //Time computing
    double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;

    //Copying data back to host, this is a blocking call and will not start until all kernels are finished
    cudaMemcpy(c, c_d, n*n*sizeof(int), cudaMemcpyDeviceToHost);
    printf("n = %dx%d \t GPU time = %fs \t CPU time = %fs\n", n, n, time_d, time_h);

    //Showing result
    // printf("Matrix C:\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     for (int j = 0; j < n; ++j)
    //     {
    //         printf("%d\t", c[i*n+j]);
    //     }
    //     printf("\n");
    // }

    // printf("Matrix C:\n");
    // for (int i = 0; i < n; ++i)
    // {
    //     for (int j = 0; j < n; ++j)
    //     {
    //         printf("%d\t", c2[i*n+j]);
    //     }
    //     printf("\n");
    // }
    //Free GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    return 0;
}
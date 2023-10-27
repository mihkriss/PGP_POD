
#include "stdio.h"
#include "stdlib.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


struct comparison {
  __device__ bool operator()(double A, double B) {
        return abs(A) < abs(B);
    }
};

__global__ void Swap(double* dev_matrix, int n, int row1, int row2) {
    int colIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int colOffset = blockDim.x * gridDim.x;
    
    for (int col = colIdx; col < 2 * n; col += colOffset) {
        int index_row1 = row1 + n * col;
        int index_row2 = row2 + n * col;
        
        double t = dev_matrix[index_row1];
        dev_matrix[index_row1] = dev_matrix[index_row2];
        dev_matrix[index_row2] = t;
    }
}

__global__ void GaussianElimination(double* dev_matrix, int n, int i, int direction) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    if (direction == 1) { 
        for (int k = idy + i + 1; k < 2 * n; k += offsety)
            for (int j = idx + i + 1; j < n; j += offsetx)
                dev_matrix[k * n + j] -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
    } else { 
        for (int k = idy + i + 1; k < 2 * n; k += offsety)
            for (int j = i - 1 - idx; j >= 0; j -= offsetx)
                dev_matrix[k * n + j] -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
    }
}

__global__ void Normal(double* dev_matrix, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    
    for (int i = idy; i < n; i += offsety) {
        for (int j = n + idx; j < 2 * n; j += offsetx) {
            int index = i + j * n;
            int diagonal_index = i + i * n;
            dev_matrix[index] /= dev_matrix[diagonal_index];
        }
    }
}

int main() {
    int n;
    scanf("%d", &n);
    double* matrix = (double*)malloc(sizeof(double) * n * n * 2);
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%lf", &matrix[i + j * n]);
    
    for (int i = 0; i < n; i++)
         for (int j = n; j < 2 * n; j++)
            matrix[i + j * n] = (i == j - n) ? 1. : 0.;


    double* dev_matrix;
    CSC(cudaMalloc(&dev_matrix, sizeof(double) * 2 * n * n));
    CSC(cudaMemcpy(dev_matrix, matrix, sizeof(double) * 2 * n * n, cudaMemcpyHostToDevice));

    comparison comp;
    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> thrust_matrix = thrust::device_pointer_cast(dev_matrix);
        
        thrust::device_ptr<double> max = thrust::max_element(&thrust_matrix[i + i * n], &thrust_matrix[n + i * n], comp);
        int max_idx = max - (thrust_matrix + i * n);
        if (max_idx != i)
            Swap<<<dim3(32), dim3(32)>>>(dev_matrix, n, i, max_idx);
        GaussianElimination<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, n, i, 1);
    }

    for (int i = n - 1; i >=0; i--)
        GaussianElimination<<<dim3(32, 32), dim3(32, 32)>>>(dev_matrix, n, i, -1);
    Normal<<<dim3(32, 32), dim3(32,32)>>>(dev_matrix, n);

    CSC(cudaMemcpy(matrix, dev_matrix, sizeof(double) * 2 * n * n, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_matrix));


    for (int i = 0; i < n; i++) {
        for (int j = n; j < 2 * n; j++) {
            printf("%.10e ", matrix[i + j * n]);
        }
        printf("\n");
    }

    free(matrix);
}


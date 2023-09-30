#include <iostream>
#include <vector>
#include <iomanip>  

using namespace std;

__global__ void findMinKernel(double* result,  double* vector1,  double* vector2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while (idx < n) {
        result[idx] = fmin(vector1[idx], vector2[idx]);
        idx += offset;
    }
}

int main() {
    int n;
    cin >> n;

    vector<double> vector1(n);
    for (int i = 0; i < n; i++) {
        cin >> vector1[i];
    }

    vector<double> vector2(n);
    for (int i = 0; i < n; i++) {
        cin >> vector2[i];
    }


    double *d_result, *d_vector1, *d_vector2;
    cudaMalloc((void**)&d_result, n * sizeof(double));
    cudaMalloc((void**)&d_vector1, n * sizeof(double));
    cudaMalloc((void**)&d_vector2, n * sizeof(double));

    cudaMemcpy(d_vector1, vector1.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector2, vector2.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    findMinKernel<<<1024, 1024>>>(d_result, d_vector1, d_vector2, n);

    vector<double> result(n);
    cudaMemcpy(result.data(), d_result, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%.10e ", result[i]);
        
    }

    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_result);

    return 0;
}

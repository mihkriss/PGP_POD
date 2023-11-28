
#include <iostream>
#include <cstdio>
#include <cstring>

#define NUM_BLOCKS 512u
#define NUM_THREADS 512u
#define CONST_MEMORY 16u
#define CONST_LOG_MEMORY 4u
#define HISTOGRAM_SIZE 25099999u

#define AVOID_CONFLICTS(n) ((n) + ((n) >> CONST_LOG_MEMORY))

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t X = call;                                           \
    if (X != cudaSuccess) {                                         \
        fprintf(stderr, "ERROR: in %s:%d. Message: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(X));         \
        exit(0);                                                    \
    }                                                               \
} while(0)

__global__ void Histogram(int* hist, int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offset) {
        atomicAdd(&hist[data[i]], 1);
    }
}

__global__ void Scan(int* data, int* prefix_sums, int size) {

	__shared__ int sharedMemory[NUM_BLOCKS];

  int step = 1;

	if (threadIdx.x < size) 
  {
		sharedMemory[(AVOID_CONFLICTS(threadIdx.x))] = data[(NUM_BLOCKS * blockIdx.x) + threadIdx.x];
		sharedMemory[(AVOID_CONFLICTS((threadIdx.x + NUM_BLOCKS / 2)))] =	data[(NUM_BLOCKS * blockIdx.x) + (threadIdx.x + NUM_BLOCKS / 2)];
	} 

	for (int i = NUM_BLOCKS / 2; i > 0; i /= 2) 
  {
		int left = step * ((threadIdx.x * 2) + 1) - 1;
		left = AVOID_CONFLICTS(left);

    int right = step * ((threadIdx.x * 2) + 2) - 1;
		right = AVOID_CONFLICTS(right);

		__syncthreads();
		if (threadIdx.x < i) {
			sharedMemory[right] += sharedMemory[left];
		}

		step *= 2;
	}

	if (threadIdx.x == 0) 
  {
		prefix_sums[blockIdx.x] = sharedMemory[AVOID_CONFLICTS(NUM_BLOCKS - 1)];
		sharedMemory[AVOID_CONFLICTS(NUM_BLOCKS - 1)] = 0;
	}

	for (int i = 1; i <= NUM_BLOCKS / 2; i *= 2) {
		step /= 2;

		int left = step * ((threadIdx.x * 2) + 1) - 1;
		left = AVOID_CONFLICTS(left);

    int right = step * ((threadIdx.x * 2) + 2) - 1;
		right = AVOID_CONFLICTS(right);

		__syncthreads();

		if (threadIdx.x < i) 
    {
			int t = sharedMemory[left];
			sharedMemory[left] = sharedMemory[right];
			sharedMemory[right] += t;
		}
	}
	__syncthreads();

	if (threadIdx.x < size) 
  {
		data[(NUM_BLOCKS * blockIdx.x) + threadIdx.x] = sharedMemory[(AVOID_CONFLICTS(threadIdx.x))];
		data[(NUM_BLOCKS * blockIdx.x) + (threadIdx.x + NUM_BLOCKS / 2)] = sharedMemory[(AVOID_CONFLICTS((threadIdx.x + NUM_BLOCKS / 2)))];
	}
}
__global__ void updateDataUsingPrefixSum(int* data, int* prefix_sums, int size) {
	if ((NUM_BLOCKS * blockIdx.x) + threadIdx.x < size) {
		data[(NUM_BLOCKS * blockIdx.x) + threadIdx.x] += prefix_sums[blockIdx.x];
	}
}

 void scan(int* dev, int size) {

  int* prefixsums;

	int count = (size / NUM_THREADS == 0) ? 1u : size / NUM_THREADS;
  
	int sharedMemory_size = NUM_THREADS * sizeof(int);

	CSC(cudaMalloc(&prefixsums, count * sizeof(int)));

	Scan<<<count, NUM_THREADS / 2, 2 * sharedMemory_size>>>(dev, prefixsums, size);
	
	if (count == 1) 
  {
		CSC(cudaFree(prefixsums));
		return;
	}

	scan(prefixsums, count);

	updateDataUsingPrefixSum<<<count, NUM_THREADS>>>(dev, prefixsums, size);

	CSC(cudaFree(prefixsums));
}

__global__ void CountSort(int* data, int* counts, int size
){
    int idx = blockDim.x * blockIdx.x +  threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += offset){
        int oldValue = i ? counts[i - 1] : 0;

        for (int j = counts[i]- 1; j >= oldValue; --j)
        {
            data[j] = i - 1;
        }
    }
}

void sort(int size, int* data) {
    int* Result;
    int* Hist;

    CSC(cudaMalloc(&Hist, size * sizeof(int)));
    CSC(cudaMalloc(&Result, HISTOGRAM_SIZE * sizeof(int)));

    CSC(cudaMemcpy(Hist, data, size * sizeof(int), cudaMemcpyHostToDevice));
    CSC(cudaMemset(Result, 0, HISTOGRAM_SIZE * sizeof(int)));

    Histogram<<<512u, 512u>>>(Result, Hist, size);
    CSC(cudaDeviceSynchronize());

    scan(Result, HISTOGRAM_SIZE);

    CountSort<<<256u, 256u>>>(Hist, Result, HISTOGRAM_SIZE);

    CSC(cudaMemcpy(data, Hist, size * sizeof(int), cudaMemcpyDeviceToHost));
}


int main() {
    int size;

    freopen(NULL, "rb", stdin);
    fread(&size, sizeof(size), 1, stdin);

    int* data = new int[size];

    fread(data, sizeof(int), size, stdin);
    fclose(stdin);
    
    sort(size, data);

    freopen(NULL, "wb", stdout);
    fwrite(data, sizeof(size), size, stdout);
    fclose(stdout);

    delete[] data;

    return 0;
}

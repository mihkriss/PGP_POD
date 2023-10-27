
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
using namespace std;

#define CSC(call)                                           \
    do {                                                    \
        cudaError_t res = call;                             \
        if (res != cudaSuccess) {                            \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
            exit(0);                                        \
        }                                                   \
    } while (0)

struct RGBColor {
    double x, y, z;
};

__device__ __host__ void RGB(RGBColor& rgb, uchar4* ps) {
    rgb = {static_cast<double>(ps->x), static_cast<double>(ps->y), static_cast<double>(ps->z)};
}

struct Point2D {
    int x, y;
};


__constant__ RGBColor constAvgs[32];
RGBColor cAvgs[32];


void CalculateAverages(vector<vector<Point2D>>&classPoints, uchar4* image, int w, int h, int nc) {
    vector<RGBColor> avgs(32, {0, 0, 0});

    for (int i = 0; i < nc; i++) {
        int np = classPoints[i].size();
        for (int j = 0; j < np; j++) {
            Point2D point = classPoints[i][j];
            uchar4 ps = image[point.y * w + point.x];
            RGBColor rgb;
            RGB(rgb, &ps);

            avgs[i].x += rgb.x;
            avgs[i].y += rgb.y;
            avgs[i].z += rgb.z;
        }

        avgs[i].x /= np;
        avgs[i].y /= np;
        avgs[i].z /= np;
    }

    for (int i = 0; i < nc; i++) {
        cAvgs[i] = avgs[i];
    }
}

__constant__ RGBColor constNormAvgs[32];
RGBColor cNormAvgs[32];


void NormalizeAverages(int nc) {
    for (int i = 0; i < nc; i++) {
        RGBColor avg = cAvgs[i];
        double magnitude = sqrt(avg.x * avg.x + avg.y * avg.y + avg.z * avg.z);
        cNormAvgs[i].x = avg.x / magnitude;
        cNormAvgs[i].y = avg.y / magnitude;
        cNormAvgs[i].z = avg.z / magnitude;
    }
}


__device__ double Similarity(uchar4 ps, int classId) {
    RGBColor rgb;
    RGB(rgb, &ps);

    RGBColor normAvg = constNormAvgs[classId];
    return rgb.x * normAvg.x + rgb.y * normAvg.y + rgb.z * normAvg.z;
}


__global__ void SpectralClustering(uchar4* out, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
  
    for (y = idy; y < h; y += offsety) {
        for (x = idx; x < w; x += offsetx) {
            double similarity = Similarity(out[y * w + x], 0);
            int cluster = 0;
            for (int i = 0; i < nc; i++) {
                double tsimilarity = Similarity(out[y * w + x], i);

                if (tsimilarity > similarity) {
                    similarity = tsimilarity;
                    cluster = i;
                }

            }
            out[y * w + x].w = (unsigned char)cluster;
        }
    }
}


int main() {
    int w, h;
    string input;
	  string output;

	  cin >> input >> output;

	  FILE* fp = fopen(input.c_str(), "rb");

    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc, np;
    cin >> nc;
    vector<vector<Point2D>> classPoints(nc);

    for (int i = 0; i < nc; i++) {
      cin >> np;
      for (int j = 0; j < np; j++) {
        Point2D point;
        cin >> point.x >> point.y;
        classPoints[i].push_back(point);
    }
}
    CalculateAverages(classPoints, data, w, h, nc);
    NormalizeAverages(nc);

    CSC(cudaMemcpyToSymbol(constAvgs, cAvgs, 32 * sizeof(RGBColor)));
    CSC(cudaMemcpyToSymbol(constNormAvgs, cNormAvgs, 32 * sizeof(RGBColor)));

    uchar4* dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    SpectralClustering << <dim3(32, 31), dim3(32, 32) >> > (dev_out, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_out));

    fp = fopen(output.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}


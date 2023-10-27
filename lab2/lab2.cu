#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <string>

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

using namespace std;

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ float grayPixel(uchar4 pixel) {
    return (0.2989f * pixel.x + 0.5870f * pixel.y + 0.1140f * pixel.z);
}

__global__ void kernel(uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    int x, y;
    for (y = idy; y < h; y += offsety)
        for (x = idx; x < w; x += offsetx) {
            uchar4 p11 = tex2D(tex, x - 1, y - 1);
            uchar4 p12 = tex2D(tex, x, y - 1);
            uchar4 p13 = tex2D(tex, x + 1, y - 1);
            uchar4 p21 = tex2D(tex, x - 1, y);
            uchar4 p22 = tex2D(tex, x, y);
            uchar4 p23 = tex2D(tex, x + 1, y);
            uchar4 p31 = tex2D(tex, x - 1, y + 1);
            uchar4 p32 = tex2D(tex, x, y + 1);
            uchar4 p33 = tex2D(tex, x + 1, y + 1);

            float p11_gray = grayPixel(p11);
            float p12_gray = grayPixel(p12);
            float p13_gray = grayPixel(p13);
            float p21_gray = grayPixel(p21);
            float p22_gray = grayPixel(p22);
            float p23_gray = grayPixel(p23);
            float p31_gray = grayPixel(p31);
            float p32_gray = grayPixel(p32);
            float p33_gray = grayPixel(p33);

            float Gx = (p13_gray + (2.0f * p23_gray) + p33_gray - p11_gray - (2.0f * p21_gray) - p31_gray);
            float Gy = (p31_gray + (2.0f * p32_gray) + p33_gray - p11_gray - (2.0f * p12_gray) - p13_gray);

            float Grad = sqrt((Gx * Gx) + (Gy * Gy));
            Grad = (Grad > 255.0f) ? 255.0f : Grad;

            out[y * w + x] = make_uchar4(static_cast<unsigned char>(Grad), static_cast<unsigned char>(Grad), static_cast<unsigned char>(Grad), p22.w);
        }
}

int main() {
    string input;
    string output;

    int w, h;
    cin >> input >> output;

    FILE* fp = fopen(input.c_str(), "rb");

    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    tex.addressMode[0] = cudaAddressModeMirror;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.channelDesc = ch;
    tex.filterMode = cudaFilterModePoint;
    tex.normalized = false;  // Use non-normalized texture coordinates

    CSC(cudaBindTextureToArray(tex, arr, ch));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<<dim3(16, 16), dim3(16, 32)>>>(dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaUnbindTexture(tex));

    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output.c_str(), "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}


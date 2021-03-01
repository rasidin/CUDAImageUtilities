/*********************************************************************
Copyright(c) 2020 LIMITGAME
Permission is hereby granted, free of charge, to any person
obtaining a copy of this softwareand associated documentation
files(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and /or sell copies of
the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :
The above copyright noticeand this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
----------------------------------------------------------------------
@file GenerateModelNormals.h
@brief Generate normals of model using vertices
@author minseob (https://github.com/rasidin)
**********************************************************************/
#include "CudaDeviceUtils.cuh"

namespace CUDAImageUtilities {
__global__ void ComputeModelNormals(float* InVertices, float* OutNormals, unsigned int InNormalNum)
{
    unsigned int normalindex = blockIdx.x * blockDim.x + threadIdx.x;
    if (normalindex >= InNormalNum) return;

    float3 v0 = make_float3(InVertices[normalindex * 3 * 3 + 0], InVertices[normalindex * 3 * 3 + 1], InVertices[normalindex * 3 * 3 + 2]);
    float3 v1 = make_float3(InVertices[normalindex * 3 * 3 + 3 + 0], InVertices[normalindex * 3 * 3 + 3 + 1], InVertices[normalindex * 3 * 3 + 3 + 2]);
    float3 v2 = make_float3(InVertices[normalindex * 3 * 3 + 6 + 0], InVertices[normalindex * 3 * 3 + 6 + 1], InVertices[normalindex * 3 * 3 + 6 + 2]);

    float3 v01 = make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
    float3 v21 = make_float3(v2.x - v1.x, v2.y - v1.y, v2.z - v1.y);

    float3 normal = cross(v01, v21);
    normal = normalize(normal);

    OutNormals[normalindex * 3 + 0] = normal.x;
    OutNormals[normalindex * 3 + 1] = normal.y;
    OutNormals[normalindex * 3 + 2] = normal.z;
}

void GenerateModelNormals(float *InVertices, float *OutNormals, const int VerticesNum) 
{
    size_t datasize_in  = VerticesNum * 3 * sizeof(float) * 3;
    size_t datasize_out = VerticesNum * sizeof(float) * 3;

    float* dIn;
    cudaMalloc(&dIn, datasize_in);
    cudaMemcpy(dIn, InVertices, datasize_in, cudaMemcpyHostToDevice);

    float* dOut;
    cudaMalloc(&dOut, datasize_out);

    dim3 threadsperblock(16, 1, 1);
    dim3 numblocks((VerticesNum + 15) / 16 * 16, 1, 1);
    ComputeModelNormals<<<numblocks, threadsperblock>>>(dIn, dOut, VerticesNum / 3);

    cudaMemcpy(OutNormals, dOut, datasize_out, cudaMemcpyDeviceToHost);

    cudaFree(dIn);
    cudaFree(dOut);
}
}
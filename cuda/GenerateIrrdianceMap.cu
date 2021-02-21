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
@file GenerateIrradianceMap.h
@brief Generate irradiance map
@author minseob (https://github.com/rasidin)
**********************************************************************/
#include "GenerateIrradianceMap.h"
#include "cuda_runtime.h"

#include "CudaDeviceUtils.cuh"

namespace CUDAImageUtilities {
texture<float4, cudaTextureType2D, cudaReadModeElementType> srctex;

__device__ float3 GenerateDirectionInTangentSpace(float2 Xi, float3 Normal)
{
    float phi = 2.0f * PI * Xi.x;
    float theta = 0.5f * PI * Xi.y;

    float3 h = make_float3(__sinf(theta) * __cosf(phi), __sinf(theta) * __sinf(phi), __cosf(theta));
    float3 up = abs(Normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f, -1.0f, 0.0f);
    float3 tanx = normalize(cross(up, Normal));
    float3 tany = cross(Normal, tanx);

    return make_float3(
        tanx.x * h.x + tany.x * h.y + Normal.x * h.z,
        tanx.y * h.x + tany.y * h.y + Normal.y * h.z,
        tanx.z * h.x + tany.z * h.y + Normal.z * h.z
    );
}

__global__ void ComputeIrradiance(float *Out, int InWidth, int InHeight, int OutWidth, int OutHeight, int SampleNum)
{
    unsigned int texx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int texy = blockIdx.y * blockDim.y + threadIdx.y;
    float2 normaluv = make_float2(((float)texx + 0.5f) / (float)OutWidth, ((float)texy + 0.5f) / (float)OutHeight);
    float3 normal = UVtoDirection(normaluv);

    float4 totallight = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int sampleidx = 0; sampleidx < SampleNum; sampleidx++) {
        float2 xi = Hammersley(sampleidx, SampleNum);
        xi.x = xi.x + (float)texx / Width * 2.0f * PI;
        float3 dir = GenerateDirectionInTangentSpace(xi, normal);
        float2 diruv = LongLatToUV(DirectionToLongLat(dir));

        float4 lightcolor = tex2D(srctex, diruv.x * (float)InWidth, diruv.y * (float)InHeight);

        float lightpower = __cosf(xi.y * 0.5f * PI);
        totallight = make_float4(totallight.x + lightcolor.x * lightpower, totallight.y + lightcolor.y * lightpower, totallight.z + lightcolor.z * lightpower, 1.0f);
    }

    Out[(texx + texy * OutWidth) * 4 + 0] = totallight.x / SampleNum;
    Out[(texx + texy * OutWidth) * 4 + 1] = totallight.y / SampleNum;
    Out[(texx + texy * OutWidth) * 4 + 2] = totallight.z / SampleNum;
    Out[(texx + texy * OutWidth) * 4 + 3] = 1.0f;
}

void GenerateIrradianceMap(float *InFloatRGBA, float *OutFloatRGBA, const int InWidth, const int InHeight, const int OutWidth, const int OutHeight, const int SampleNum)
{
    size_t datasize_in  = InWidth * InHeight * sizeof(float) * 4;
    size_t datasize_out = OutWidth * OutHeight * sizeof(float) * 4;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, InWidth, InHeight);
    cudaMemcpyToArray(cuArray, 0, 0, InFloatRGBA, datasize_in, cudaMemcpyHostToDevice);

    srctex.addressMode[0] = cudaAddressModeWrap;
    srctex.addressMode[1] = cudaAddressModeWrap;
    srctex.filterMode = cudaFilterModeLinear;
    srctex.normalized = false;

    cudaBindTextureToArray(srctex, cuArray, channelDesc);

    float* dOut;
    cudaMalloc(&dOut, datasize_out);

    dim3 threadsperblock(16, 16);
    dim3 numblocks(OutWidth / threadsperblock.x, OutHeight / threadsperblock.y);
    ComputeIrradiance<<<numblocks, threadsperblock>>>(dOut, InWidth, InHeight, OutWidth, OutHeight, SampleNum);

    cudaMemcpy(OutFloatRGBA, dOut, datasize_out, cudaMemcpyDeviceToHost);

    cudaFreeArray(cuArray);
    cudaFree(dOut);
}
}
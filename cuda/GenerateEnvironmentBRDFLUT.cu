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
@file GenerateEnvironmentBRDFLUT.h
@brief Generate Environment BRDF LUT
@author minseob (leeminseob@outlook.com)
**********************************************************************/
#include "GenerateEnvironmentBRDFLUT.h"
#include "cuda_runtime.h"

#include "CudaDeviceUtils.cuh"

/** Reference from 'Real Shading in Unreal Engine 4 by Brian Karis' */
namespace CUDAImageUtilities {
/** Reference from 'Real Shading in Unreal Engine 4 by Brian Karis' */
__device__ float3 ImportanceSampleGGX(float2 Xi, float Roughness, float3 Normal)
{
    float a = Roughness * Roughness;

    float phi = 2.0f * PI * Xi.x;
    float costheta = sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    float sintheta = sqrt(1.0f - costheta * costheta);

    float3 h = make_float3(sintheta * cos(phi), sintheta * sin(phi), costheta);

    float3 up = abs(Normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tanx = normalize(cross(up, Normal));
    float3 tany = cross(Normal, tanx);

    return make_float3(
        tanx.x * h.x + tany.x * h.y + Normal.x * h.z,
        tanx.y * h.x + tany.y * h.y + Normal.y * h.z,
        tanx.z * h.x + tany.z * h.y + Normal.z * h.z
    );
}

__device__ float2 IntergrateBRDF(float Roughness, float NoV, int SampleNum)
{
    float3 v = make_float3(sqrt(1.0f - NoV * NoV), 0, NoV);

    float a = 0.0f;
    float b = 0.0f;
    for (unsigned int i = 0; i < SampleNum; i++) {
        float2 xi = Hammersley(i, SampleNum);
        float3 h = ImportanceSampleGGX(xi, Roughness, make_float3(0.0f, 0.0f, 1.0f));
        float voh = v.x * h.x + v.y * h.y + v.z * h.z;
        float3 l = make_float3(2.0f * voh * h.x - v.x, 2.0f * voh * h.y - v.y, 2.0f * voh * h.z - v.z);

        float nol = saturate(l.z);
        float noh = saturate(h.z);
        voh = saturate(voh);

        if (nol > 0)
        {
            float g = G_Smith(Roughness, NoV, nol);
            float g_vis = g * voh / (noh * NoV);
            float fc = (1 - voh) * (1 - voh) * (1 - voh) * (1 - voh) * (1 - voh);
            a += (1 - fc) * g_vis;
            b += fc * g_vis;
        }
    }

    return make_float2(a / SampleNum, b / SampleNum);
}

__global__ void ComputeEnvironmentBRDFLUT(float* Out, int Width, int Height, int SampleNum)
{
    unsigned int texx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int texy = blockIdx.y * blockDim.y + threadIdx.y;

    float nov = max((float)texx / (float)Width, 0.0001f);
    float roughness = (float)texy / (float)Height;
    float2 AB = IntergrateBRDF(roughness, nov, SampleNum);

    Out[(texx + texy * Width) * 2 + 0] = AB.x;
    Out[(texx + texy * Width) * 2 + 1] = AB.y;
}

void GenerateEnvironmentBRDFLUT(float* OutFloatRG, const int Width, const int Height, const int SampleNum)
{
    size_t datasize_out = Width * Height * sizeof(float) * 2;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, Width, Height);

    float* dOut;
    cudaMalloc(&dOut, datasize_out);

    dim3 threadsperblock(16, 16);
    dim3 numblocks(Width / threadsperblock.x, Height / threadsperblock.y);
    ComputeEnvironmentBRDFLUT << <numblocks, threadsperblock >> > (dOut, Width, Height, SampleNum);

    cudaMemcpy(OutFloatRG, dOut, datasize_out, cudaMemcpyDeviceToHost);
    cudaFree(dOut);
}
}
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
@file GeneratePrefilteredReflectionMap.cu
@brief Generate prefiltered reflection map
@author minseob (https://github.com/rasidin)
**********************************************************************/
#include "GeneratePrefilteredReflectionMap.h"
#include "cuda_runtime.h"

#include "CudaDeviceUtils.cuh"

namespace CUDAImageUtilities {
#define ROUGHNEST_MIP 6 // 64x32 is diffuse
texture<float4, cudaTextureType2D, cudaReadModeElementType> srctex;

/** Reference from 'Real Shading in Unreal Engine 4 by Brian Karis' */
__device__ float3 PRM_ImportanceSampleGGX(float2 Xi, float Roughness, float3 Normal)
{
    float a = Roughness * Roughness;

    float phi = 2.0f * PI * Xi.x;
    float costheta = sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    float sintheta = sqrt(1.0f - costheta * costheta);

    float3 h = make_float3(sintheta * cos(phi), sintheta * sin(phi), costheta);

    float3 up = abs(Normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(0.0f,-1.0f, 0.0f);
    float3 tanx = normalize(cross(up, Normal));
    float3 tany = cross(Normal, tanx);

    return make_float3(
        tanx.x * h.x + tany.x * h.y + Normal.x * h.z,
        tanx.y * h.x + tany.y * h.y + Normal.y * h.z,
        tanx.z * h.x + tany.z * h.y + Normal.z * h.z
    );
}

__device__ float3 PrefilterEnvMap(float Roughness, float3 R, int Width, int Height, int SampleNum)
{
    float3 n = R;
    float3 v = R;

    float3 prefilteredcolor = make_float3(0.0f, 0.0f, 0.0f);
    float totalweight = 0.0f;
    for (int i = 0; i < SampleNum; i++) {
        float2 xi = Hammersley(i, SampleNum);
        float3 h = PRM_ImportanceSampleGGX(xi, Roughness, n);
        float3 l = make_float3(2.0f * dot(v, h) * h.x - v.x, 2.0f * dot(v, h) * h.y - v.y, 2.0f * dot(v, h) * h.z - v.z);

        float nol = saturate(dot(n, l));
        if (nol > 0.0f)
        {
            float2 longlat = DirectionToLongLat(l);
            float2 longlatUV = make_float2(longlat.x / 2.0f / PI, longlat.y / PI);
            if (longlatUV.y < 0.0f) {
                longlatUV.y += 1.0f;
                longlatUV.x += PI;
            }
            if (longlatUV.y > 1.0f) {
                longlatUV.y -= 1.0f;
                longlatUV.x += PI;
            }
            if (longlatUV.x < 0.0f)
                longlatUV.x += 1.0f;
            if (longlatUV.x > 1.0f)
                longlatUV.x -= 1.0f;
            float4 envcolor = tex2D(srctex, longlatUV.x * (float)Width, longlatUV.y * (float)Height);
            prefilteredcolor = make_float3(prefilteredcolor.x + envcolor.x * nol, prefilteredcolor.y + envcolor.y * nol, prefilteredcolor.z + envcolor.z * nol);
            totalweight = totalweight + nol;
        }
    }
    if (totalweight == 0.0f)
        return make_float3(1.0f, 0.0f, 0.0f);

    return make_float3(prefilteredcolor.x / totalweight, prefilteredcolor.y / totalweight, prefilteredcolor.z / totalweight);
}

__global__ void ComputePrefilteredReflectionMap(float *Out, int InWidth, int InHeight, int OutWidth, int OutHeight, int SampleNum)
{
    unsigned int texx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int texy = blockIdx.y * blockDim.y + threadIdx.y;
    float2 normaluv = make_float2(((float)texx + 0.5f) / (float)OutWidth, ((float)texy + 0.5f) / (float)OutHeight);
    float roughness = min(1.0f, powf(2, ROUGHNEST_MIP - log2f(OutWidth)));

    float3 prefilteredcolor = PrefilterEnvMap(roughness, UVtoDirection(normaluv), InWidth, InHeight, SampleNum);

    Out[(texx + texy * OutWidth) * 4 + 0] = prefilteredcolor.x;
    Out[(texx + texy * OutWidth) * 4 + 1] = prefilteredcolor.y;
    Out[(texx + texy * OutWidth) * 4 + 2] = prefilteredcolor.z;
    Out[(texx + texy * OutWidth) * 4 + 3] = 1.0f;
}

void GeneratePrefilteredReflectionMap(float* InFloatRGBA, float* OutFloatRGBA, const int InWidth, const int InHeight, const int OutWidth, const int OutHeight, const int MipCount, const int SampleNum)
{
    size_t datasize_in = InWidth * InHeight * sizeof(float) * 4;
    size_t datasize_out = 0u;

    for (int mip = 0; mip < MipCount; mip++) {
        datasize_out += (OutWidth >> mip) * (OutHeight >> mip) * sizeof(float) * 4;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, InWidth, InHeight);
    cudaMemcpyToArray(cuArray, 0, 0, InFloatRGBA, datasize_in, cudaMemcpyHostToDevice);

    srctex.addressMode[0] = cudaAddressModeWrap;
    srctex.addressMode[1] = cudaAddressModeWrap;
    srctex.filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(srctex, cuArray, channelDesc);

    float* dOut = nullptr;
    cudaMalloc(&dOut, datasize_out);

    float* dOutMip = dOut;

    for (int mip = 0; mip < MipCount; mip++) {
        dim3 threadsperblock(16, 16);
        int mipwidth = OutWidth >> mip;
        int mipheight = OutHeight >> mip;

        dim3 numblocks(mipwidth / threadsperblock.x, mipheight / threadsperblock.y);
        ComputePrefilteredReflectionMap<<<numblocks, threadsperblock>>>(dOutMip, InWidth, InHeight, mipwidth, mipheight, SampleNum);

        dOutMip += mipwidth * mipheight * 4;
    }

    cudaMemcpy(OutFloatRGBA, dOut, datasize_out, cudaMemcpyDeviceToHost);

    cudaFreeArray(cuArray);
    cudaFree(dOut);
}
}

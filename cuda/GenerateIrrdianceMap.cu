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
@author minseob (leeminseob@outlook.com)
**********************************************************************/
#include "GenerateIrradianceMap.h"
#include "cuda_runtime.h"

#define PI 3.141592654f

namespace CUDAImageUtilities {
texture<float4, cudaTextureType2D, cudaReadModeElementType> srctex;

__device__ float3 UVtoDirection(float2 uv)
{
    float2 longlat = make_float2((2.0f * uv.x - 1.0f) * PI, (0.5f - uv.y) * PI);
    return make_float3(cos(longlat.y) * sin(longlat.x), sin(longlat.y) * sin(longlat.x), cos(longlat.y));
}

__device__ float2 Hammersley(unsigned int i, unsigned int n)
{
    return make_float2(float(i) / float(n), float(__brev(i)) * 2.328306465386963e-10);
}

__global__ void ComputeIrradiance(float *Out, int InWidth, int InHeight, int OutWidth, int OutHeight, int SampleNum)
{
    unsigned int texx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int texy = blockIdx.y * blockDim.y + threadIdx.y;
    float2 normaluv = make_float2(((float)texx + 0.5f) / (float)OutWidth, ((float)texy + 0.5f) / (float)OutHeight);

    float4 totallight = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int sampleidx = 0; sampleidx < SampleNum; sampleidx++) {
        float2 hammersleylonglat = Hammersley(sampleidx, SampleNum);
        float2 normalphiandtheta = make_float2(normaluv.x * 2.0f * PI, acos(1.0f - 2.0f * normaluv.y));
        float2 randomdirection = make_float2(normalphiandtheta.x + hammersleylonglat.x * 2.0f * PI, normalphiandtheta.y + (0.5f * hammersleylonglat.y) * PI);
        if (randomdirection.y < 0.0f) {
            randomdirection.y = abs(randomdirection.y);
            randomdirection.x += PI;
        }
        else if (randomdirection.y > PI) {
            randomdirection.y = 2.0f * PI - randomdirection.y;
            randomdirection.x += PI;
        }
        if (randomdirection.x > 2.0f * PI) {
            randomdirection.x -= 2.0f * PI;
        }
        else if (randomdirection.x < 0.0f) {
            randomdirection.x += 2.0f * PI;
        }
        float2 randomuv = make_float2(randomdirection.x / (2.0f * PI), -cos(randomdirection.y) * 0.5f + 0.5f);

        float4 lightcolor = tex2D(srctex, randomuv.x * (float)InWidth, randomuv.y * (float)InHeight);

        float lightpower = (1.0f - hammersleylonglat.y) * 2.0f;
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

    cudaFree(dOut);
}
}
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
@file CudaDeviceUtils.h
@brief Utility for cuda computing
@author minseob (leeminseob@outlook.com)
**********************************************************************/
#ifndef CUDAIMAGEUTILITIES_CUDADEVICEUTILS_H_
#define CUDAIMAGEUTILITIES_CUDADEVICEUTILS_H_

#define PI 3.141592654f

namespace CUDAImageUtilities {
inline __device__ float3 UVtoDirection(float2 uv)
{
    float2 longlat = make_float2((2.0f * uv.x - 1.0f) * PI, (0.5f - uv.y) * PI);
    return make_float3(cos(longlat.y) * sin(longlat.x), sin(longlat.y) * sin(longlat.x), cos(longlat.y));
}
inline __device__ float2 DirectionToLongLat(float3 dir)
{
    return make_float2(atan2f(dir.x, dir.z), acos(dir.y));
}
inline __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline __device__ float3 normalize(float3 n)
{
    float length = n.x * n.x + n.y * n.y + n.z * n.z;
    return (length > 0) ? make_float3(n.x / length, n.y / length, n.z / length) : make_float3(0.0f, 0.0f, 0.0f);
}
inline __device__ float2 Hammersley(unsigned int i, unsigned int n)
{
    return make_float2(float(i) / float(n), float(__brev(i)) * 2.328306465386963e-10);
}
inline __device__ float G_Smith(float Roughness, float nov, float nol)
{
    float nov2 = nov * nov;
    float nol2 = nol * nol;
    float g_nov = 2.0f * nov / (nov + sqrt(nov2 + Roughness * Roughness * (1.0f - nov2)));
    float g_nol = 2.0f * nol / (nol + sqrt(nol2 + Roughness * Roughness * (1.0f - nol2)));
    return g_nov * g_nol;
}
}

#endif // CUDAIMAGEUTILITIES_CUDADEVICEUTILS_H_
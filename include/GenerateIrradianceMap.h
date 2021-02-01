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
@brief Generate irradiance map (lambertian reflection)
@author minseob (leeminseob@outlook.com)
**********************************************************************/
#ifndef CUDAIMAGEUTILITIES_GENERATEIRRADIANCEMAP_H_
#define CUDAIMAGEUTILITIES_GENERATEIRRADIANCEMAP_H_

namespace CUDAImageUtilities {
void GenerateIrradianceMap(float *InFloatRGBA, float *OutFloatRGBA, const int InWidth, const int InHeight, const int OutWidth, const int OutHeight, const int SampleNum);
}

#endif // CUDAIMAGEUTILITIES_GENERATEIRRADIANCEMAP_H_
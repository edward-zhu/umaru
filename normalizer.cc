extern "C" {
#include <luaT.h>
#include <TH/TH.h>
}

#include <cmath>
#include <iostream>
#include <vector>

const double SIGMA_RATE_VERT	= 0.5;
const double SIGMA_RATE_HORZ 	= 1.0;
const double SIGMA_RATE_CENTER	= 0.3;
const double RANGE_RATE			= 4.0;
const long TARGET_HEIGHT		= 64;

template <class T>
static void gauss1dWithMask(double * out, T  * in, long size, double * mask, long mask_size) {
    // apply it
	long range = (mask_size - 1) / 2;
    int n = size;
    for (int i = 0; i < n; i++) {
        double total = 0.0;
        for (int j = 0; j < mask_size; j++) {
            int index = i+j-range;
            if (index < 0)
                index = 0;
            if (index >= n)
                index = n-1;
            total += in[index] * mask[j];
        }
        out[i] = double(total);
    }
}

static void create1DMask(double * & mask, double sigma, long & mask_size) {
    int range = 1 + int(3.0*sigma);
	mask_size = 2 * range + 1;	
	mask = new double[mask_size];
    for (int i = 0; i <= range; i++) {
		double sd = sigma * sigma;
        double y = exp(-i*i/2.0/sd);
        mask[range+i] = mask[range-i] = y;
    }
    double total = 0.0;
    for (int i = 0; i < mask_size; i++)
        total += mask[i];
    for (int i = 0; i < mask_size; i++) {
    	mask[i] /= total;
    }
	
	
}

template <class T>
static void gauss1d(double * out, T  * in, long size, double sigma) {
    double * mask = NULL;
	long ms;
	create1DMask(mask, sigma, ms);
    gauss1dWithMask(out, in, size, mask, ms);
	delete [] mask;
}

static void getDim1(double * in, double * out, long w, long h, long index) {
	for (int i = 0; i < h; i++) {
		out[i] = in[i * w + index];
	}
}

static void setDim1(double * in, double * out, long w, long h, long index) {
	for (int i = 0; i < h; i++) {
		out[i * w + index] = in[i];
	}
}

static void gauss2d(double * src, long w, long h, double sigmaX, double sigmaY) {
	double tmp[h];
	double in_tmp[h];
	double * maskX = NULL, * maskY = NULL;
	long msX, msY;
	
	create1DMask(maskY, sigmaY, msY);
	create1DMask(maskX, sigmaX, msX);

	
	for (int i = 0; i < w; i++) {
		getDim1(src, in_tmp, w, h, i);
		gauss1dWithMask(tmp, in_tmp, h, maskY, msY);
		setDim1(tmp, src, w, h, i);
	}
	
	double tmp2[w];
	for (int i = 0; i < h; i++) {
		memcpy(tmp2, src + w * i, w * sizeof(double));
		gauss1dWithMask(src + w * i, tmp2, w, maskX, msX);
	}
	
	delete [] maskX;
	delete [] maskY;
}



static double bilinear(double * in, int w, int h, double x, double y) {
	int xi = int(x), yi = int(y), xt = xi + 1, yt = yi + 1;
	double xf = x - xi, yf = y - yi;
	
	// printf("(%d, %d)\n", xi, yi);
	
	if (xi > w - 1 || yi > h - 1 || x < 0 || y < 0) {
		return 0;
	}
	
	xi = xi < 0 ? 0 : xi;
	yi = yi < 0 ? 0 : yi;
	
	
	xi = xi > w - 1 ? w - 1 : xi;
	yi = yi > h - 1 ? h - 1 : yi;
	
	
	xt = xt > w - 1 ? w - 1 : xt;
	yt = yt > h - 1 ? h - 1 : yt;
	
	
	
	double p00 = in[yi * w + xi];
	double p01 = in[yt * w + xi];
	double p10 = in[yi * w + xt];
	double p11 = in[yt * w + xt];
	
	double result = p00 * (1.0 - xf) * (1.0 - yf) + p10 * xf * (1.0 - yf) + p01 * (1.0 - xf) * yf + p11 * xf * yf;
	if (result < 0) {
		printf("warning result < 0. %.4lf, %.4lf, %.4f, %.4f\n" \
			 		"%.4f, %.4f, %.4f, %.4f\n", x, y, xf, yf, p00, p01, p10, p11);
	}
	
	return result;
}

static void measure(THDoubleTensor * src, double * & center, double & mean, int & r) {
	long h = src->size[0];
	long w = src->size[1];
	double sigmaX = h * SIGMA_RATE_HORZ;
	double sigmaY = h * SIGMA_RATE_VERT;
	double * dataSrc = THDoubleTensor_data(src);
	THDoubleTensor * smooth = THDoubleTensor_newClone(src);
	double * dataSmooth = THDoubleTensor_data(smooth);
	gauss2d(dataSmooth, w, h, sigmaX, sigmaY);
	
	THDoubleTensor * minVT = THDoubleTensor_new();
	THLongTensor * minT = THLongTensor_new();
	
	THDoubleTensor_max(minVT, minT, smooth, 0);
	
	long * min = THLongTensor_data(minT);
	
	center = new double[w];
	
	gauss1d(center, min, w, h * SIGMA_RATE_CENTER);
	
	double s1 = 0.0, sy = 0.0;
	
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			s1 += dataSrc[i * w + j];
			sy += dataSrc[i * w + j] * fabs(i - center[j]);
		}
	}
	
	
	mean = sy / s1;
	r = int(mean * RANGE_RATE + 1);
	
	/* printf("mean = %lf r = %d\n", mean, r); */
}

static void normalize
	(THDoubleTensor * src, THDoubleTensor * out, double * center, double mean, int r) {
	long h = src->size[0];
	long w = src->size[1];
	int target_height = TARGET_HEIGHT;
	float scale = (2.0 * r) / TARGET_HEIGHT;
	int target_width = fmax(int(w / scale), 1);
	
	double * inData = THDoubleTensor_data(src);
	
	THDoubleTensor_resize2d(out, target_height, target_width);
	
	// printf("scale = %.4f\n", scale);
	
	double * outData = THDoubleTensor_data(out);
	
	
	for (int i = 0; i < target_height; i++) {
		for (int j = 0; j < target_width; j++) {
			float x = scale * j;
			float y = scale * (i - target_height / 2) + center[int(x)];
			// printf(" = %d\n", (i - target_height / 2));
			outData[i * target_width + j] = bilinear(inData, w, h, x, y);
		}
	}
	
}

static int normalizer_gauss1d(lua_State * L)
{
	THDoubleTensor * input = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	double sigma = luaL_checknumber(L, 2);
	int size = input->size[0];
	
	printf("sigma = %.4lf\n", sigma);
	
	double * data = THDoubleTensor_data(input);
	
	THDoubleTensor * outputT = THDoubleTensor_newClone(input);
	
	double * output = THDoubleTensor_data(outputT);
	
	gauss1d(output, data, size, sigma);
	
	return 0;
}

static int normalizer_gauss2d(lua_State * L)
{
	THDoubleTensor * input = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	double sigmaX = luaL_checknumber(L, 2);
	double sigmaY = luaL_checknumber(L, 3);
	long h = input->size[0];
	long w = input->size[1];
	
	double * data = THDoubleTensor_data(input);
	
	printf("w = %ld, h = %ld\n", w, h);
	printf("sigmaX = %.4lf sigmaY = %.4lf\n", sigmaX, sigmaY);
	
	gauss2d(data, w, h, sigmaX, sigmaY);
	
	return 0;
}

static int normalizer_normalize(lua_State * L)
{
	THDoubleTensor * input = 
		(THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	THDoubleTensor * output = 
		(THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	
	double * center = NULL;
	double mean = 0.0;
	int r = 0;
	
	measure(input, center, mean, r);
	normalize(input, output, center, mean, r);
	
	
	delete [] center;
	return 0;
}

static const struct luaL_reg normalizer[] = {
	{"gauss1d", normalizer_gauss1d},
	{"gauss2d", normalizer_gauss2d},
	{"normalize", normalizer_normalize},
	{NULL, NULL}
};

LUA_EXTERNC int luaopen_normalizer(lua_State *L) {
	luaL_openlib(L, "normalizer", normalizer, 0);
	return 1;
}
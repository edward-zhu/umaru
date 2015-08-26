extern "C" {
#include <luaT.h>
#include <TH/TH.h>
#include <assert.h>
}

#include <cmath>
#include <iostream>
#include <vector>

#define ENABLE_OPENMP

static const float EXP_MAX		= 1e10;
static const float EXP_MIN		= 1e-10;
static const float LOG_ZERO		= -1e10;
static const float LOG_INF		= 1e10;
static const float EXP_LIMIT	= log(EXP_MAX);

static float inline fmax(float x, float y) {
	return (x > y) ? x : y;
}

static float safe_log(float x) {
	if (x == 0) {
		return LOG_ZERO;
	}
	else if (x > 0) {
		return log(x);
	}
	else {
		perror("Error: passing a negative number to the log function.");
		return LOG_ZERO;
	}
}

static float safe_exp(float x) {
	if (x == LOG_ZERO) {
		return 0;
	}
	if (x >= EXP_LIMIT) {
		return EXP_MAX;
	}
	return exp(x);
}

static float log_add(float x, float y) {
	if (fabs(x - y) > 10) {
		fmax(x, y);
	}
	
	if (x < y) {
		return y + log(1.0 + safe_exp(x - y));
	}
	return x + log(1.0 + safe_exp(y - x));
}

static float log_sub(float x, float y) {
	if (y == LOG_ZERO) {
		return x;
	}
	if (y >= x) {
		return LOG_ZERO;
	}
	return x + log(1.0 - safe_exp(y - x));
}

static float log_mul(float x, float y) {
	if (y == LOG_ZERO or x == LOG_ZERO) {
		return LOG_ZERO;
	}
	
	return x + y;
}

static THFloatTensor * __get_forward_variable(THFloatTensor * outputTable, THFloatTensor * alignedTable, THFloatTensor * targetT) {
	int T = outputTable->size[0];
	int L = targetT->size[0];
	
	float * aligned = THFloatTensor_data(alignedTable);
	float * target = THFloatTensor_data(targetT);
	
	
	THFloatTensor * fvsT = THFloatTensor_newWithSize2d(T, L);
	THFloatStorage_fill(fvsT->storage, LOG_ZERO);
	float * fvs = THFloatTensor_data(fvsT);
	
	fvs[0] = aligned[0];
	fvs[1] = aligned[1];
	
	int lower_bound = -1, upper_bound = 2;
	
	float fvs_tmp, fvs_i1u, fvs_i1u1, fvs_i1u2;
	
	for(int i = 1; i < T; i++) {
		// adjust bounds, some positions would never been visited
		
		upper_bound += 2;
		if (upper_bound > L) {
			upper_bound = L;
		}
		
		lower_bound = L - 2 * (T - i);
		if (lower_bound < 1) {
			lower_bound = 1;
		}

		assert(lower_bound >= 0 && lower_bound < T * L);
		assert(upper_bound >= 0 && upper_bound < T * L);

		for (int u = lower_bound; u < upper_bound; u++) {
			float tmp = LOG_ZERO;
			
			fvs_i1u = fvs[(i - 1) * L + u];
			fvs_i1u1 = (u > 0) ? fvs[(i - 1) * L + u - 1] : LOG_ZERO;
			fvs_i1u2 = (u > 1 && target[u - 2] != target[u]) ? fvs[(i - 1) * L + u - 2] : LOG_ZERO;
			
			if (u % 2) {
				tmp = log_add(tmp, fvs_i1u);
				tmp = log_add(tmp, fvs_i1u1);
				tmp = log_add(tmp, fvs_i1u2);	
			}
			else {
				tmp = log_add(tmp, fvs_i1u);
				tmp = log_add(tmp, fvs_i1u1);
			}
			fvs[i * L + u] = log_mul(tmp, aligned[i * L + u]);
		}
		
	}
	return fvsT;
}

static THFloatTensor * __get_backward_variable(THFloatTensor * outputTable, THFloatTensor * alignedTable, THFloatTensor * targetT) {
	int T = outputTable->size[0];
	int L = targetT->size[0];
	
	float * aligned = THFloatTensor_data(alignedTable);
	float * target = THFloatTensor_data(targetT);
	
	THFloatTensor * bvsT = THFloatTensor_newWithSize2d(T, L);
	THFloatStorage_fill(bvsT->storage, LOG_ZERO);
	float * bvs = THFloatTensor_data(bvsT);
	
	assert(T * L >= 2);

	bvs[T * L - 1] = 0;
	bvs[T * L - 2] = 0;
	
	int lower_bound = -1, upper_bound = L - 3;
	
	float bvs_tmp, bvs_i1u, bvs_i1u1, bvs_i1u2;
	
	
	for(int i = T - 2; i >= 0; i--) {
		// adjust bounds, some positions would never been visited
		
		upper_bound -= 2;
		if (upper_bound < 0) {
			upper_bound = 0;
		}
		
		lower_bound = 2 * i + 1;
		if (lower_bound > L - 2) {
			lower_bound = L - 2;
		}
		
		if (lower_bound < 0) {
			lower_bound = 0;
		}
		
		if (upper_bound > L - 2) {
			upper_bound = L - 2;
		}

		assert(upper_bound >= 0 && upper_bound < L);
		assert(lower_bound >= 0 && lower_bound < L);
		
		// printf("%d  %d\n", upper_bound, lower_bound);
		
		for (int u = lower_bound; u >= upper_bound; u--) {
			
			float tmp = LOG_ZERO;
			
			assert((i * L + u < T * L) && (i * L + u) >= 0);
			assert(((i + 1) * L + u) >= 0 && ((i + 1) * L + u) < T * L);
			assert(((i + 1) * L + u + 1) >= 0 && ((i + 1) * L + u + 1) < T * L);
			assert((u >= L - 1) || ((i + 1) * L + u + 1 >= 0 && ((i + 1) * L + u + 1 < T * L)));
			assert(!(u < L - 2 && target[u + 2] != target[u]) || ((i + 1) * L + u + 2) >= 0 && ((i + 1) * L + u + 2) < T * L);

			bvs_i1u = bvs[(i + 1) * L + u];
			bvs_i1u1 = (u < L - 1) ? bvs[(i + 1) * L + u + 1] : LOG_ZERO;
			bvs_i1u2 = (u < L - 2 && target[u + 2] != target[u]) ? bvs[(i + 1) * L + u + 2] : LOG_ZERO;
			
			tmp = log_mul(aligned[(i + 1) * L + u], bvs_i1u);
			tmp = log_add(tmp, log_mul(aligned[(i + 1) * L + u + 1], bvs_i1u1));
			
			if (u % 2 && u < L - 2) {
				assert(((i + 1) * L + u + 2) >= 0 && ((i + 1) * L + u + 2) < T * L);
				tmp = log_add(tmp, log_mul(aligned[(i + 1) * L + u + 2], bvs_i1u2));		
			}
			
			bvs[i * L + u] = tmp;
			
			if ((i + 1) * L + u + 1 >= T * L) {
				perror("out of range\n");
			}
		}
		
	}
	return bvsT;
}


static THFloatTensor * __get_grad(THFloatTensor * fbT, THFloatTensor * outputTable, THFloatTensor * targetT, float pzx) {
	
	int T = fbT->size[0];
	int L = targetT->size[0];
	int class_num = outputTable->size[1];
	
	int pos;
	
	THFloatTensor * gradT = THFloatTensor_newWithSize2d(T, class_num);
	float * fb = THFloatTensor_data(fbT);
	float * output = THFloatTensor_data(outputTable);
	float * grad = THFloatTensor_data(gradT);
	float * target = THFloatTensor_data(targetT);
	
	float tmp_sum = 0, u = 0, tmp = 0;

	int t;

	#ifdef ENABLE_OPENMP
	#pragma omp parallel for private(tmp_sum, u, tmp, pos) lastprivate(t)
	#endif
	for (t = 0; t < T; t++) {
		// printf("%d\n", t);
		
		for (int k = 0; k < class_num; k++) {
			pos = t * class_num + k;

			assert(pos >=0 && pos < class_num * T);

			tmp_sum = LOG_ZERO;
			tmp = log_mul(-pzx, -output[pos]);
			u = k + 1;
			
			if (u == class_num) {
				u = 0;
			}
			
			for (int i = 0; i < L; i++) {
				if (target[i] == u) {
					// printf("%.4f\n", fb[t * L + i]);

					assert((t * L + i) >=0 && (t * L + i) < T * L);

					tmp_sum = log_add(fb[t * L + i], tmp_sum);
				}
			}
			
			
			tmp = log_mul(tmp, tmp_sum);
			
			grad[pos] = -safe_exp(tmp);
		}
	}
	#pragma omp barrier
	
	return gradT;
}

static int ctc_get_forward_variable(lua_State * L) {
	THFloatTensor * output = (THFloatTensor *)luaT_checkudata(L, 1, "torch.FloatTensor");
	THFloatTensor * alignedTable = (THFloatTensor *)luaT_checkudata(L, 2, "torch.FloatTensor");
	THFloatTensor * target = (THFloatTensor *)luaT_checkudata(L, 3, "torch.FloatTensor");
	
	
	
	THFloatTensor * fvs = __get_forward_variable(output, \
													alignedTable, target);
	
												
	luaT_pushudata(L, fvs, "torch.FloatTensor");
	
	return 1;											
}

static int ctc_get_backward_variable(lua_State * L) {
	THFloatTensor * output = (THFloatTensor *)luaT_checkudata(L, 1, "torch.FloatTensor");
	THFloatTensor * alignedTable = (THFloatTensor *)luaT_checkudata(L, 2, "torch.FloatTensor");
	THFloatTensor * target = (THFloatTensor *)luaT_checkudata(L, 3, "torch.FloatTensor");
	
	
	
	THFloatTensor * bvs = __get_backward_variable(output, \
													alignedTable, target);
	
	/*
	float * data = THFloatTensor_data(bvs);
	
	for (int i = 0; i < bvs->size[0]; i++) {
		for (int j = 0; j < bvs->size[1]; j++) {
			printf("%.4f\t", data[i * bvs->size[1] + j] == -1e10 ? 0 : data[i * bvs->size[1] + j]);
		}
		printf("\n");
	}
	*/
									
	luaT_pushudata(L, bvs, "torch.FloatTensor");
	
	return 1;											
}



static int ctc_get_grad(lua_State * L) {
	THFloatTensor * fb = (THFloatTensor *)luaT_checkudata(L, 1, "torch.FloatTensor");
	THFloatTensor * outputTable = (THFloatTensor *)luaT_checkudata(L, 2, "torch.FloatTensor");
	THFloatTensor * target = (THFloatTensor *)luaT_checkudata(L, 3, "torch.FloatTensor");
	float pzx = luaL_checknumber(L, 4);
	
	THFloatTensor * grad = __get_grad(fb, outputTable, target, pzx);
	
	luaT_pushudata(L, grad, "torch.FloatTensor");
	
	return 1;
}


static const struct luaL_reg libctc[] = {
	{"get_forward_variable", ctc_get_forward_variable},
	{"get_backward_variable", ctc_get_backward_variable},
	{"get_grad", ctc_get_grad},
	{NULL, NULL}
};

LUA_EXTERNC int luaopen_libctc(lua_State *L) {
	luaL_openlib(L, "libctc", libctc, 0);
	return 1;
}
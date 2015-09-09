extern "C" {
#include <luaT.h>
#include <TH/TH.h>
#include <assert.h>
}

#include <cmath>
#include <iostream>
#include <vector>

// #define ENABLE_OPENMP

static const double EXP_MAX		= 1e100;
static const double EXP_MIN		= 1e-100;
static const double LOG_ZERO	= -1e100;
static const double LOG_INF		= 1e100;
static const double EXP_LIMIT	= log(EXP_MAX);


static double safe_log(double x) {
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

static double safe_exp(double x) {
	if (x == LOG_ZERO) {
		return 0;
	}
	if (x >= EXP_LIMIT) {
		return EXP_MAX;
	}
	return exp(x);
}

static double log_add(double x, double y) {
	if (fabs(x - y) > 10) {
		fmax(x, y);
	}
	
	if (x < y) {
		return y + log(1.0 + safe_exp(x - y));
	}
	return x + log(1.0 + safe_exp(y - x));
}

static double log_sub(double x, double y) {
	if (y == LOG_ZERO) {
		return x;
	}
	if (y >= x) {
		return LOG_ZERO;
	}
	return x + log(1.0 - safe_exp(y - x));
}

static double log_mul(double x, double y) {
	if (y == LOG_ZERO or x == LOG_ZERO) {
		return LOG_ZERO;
	}
	
	return x + y;
}

static THDoubleTensor * __get_forward_variable(THDoubleTensor * outputTable, THDoubleTensor * alignedTable, THDoubleTensor * targetT) {
	int T = outputTable->size[0];
	int L = targetT->size[0];
	
	double * aligned = THDoubleTensor_data(alignedTable);
	double * target = THDoubleTensor_data(targetT);
	
	
	THDoubleTensor * fvsT = THDoubleTensor_newWithSize2d(T, L);
	THDoubleStorage_fill(fvsT->storage, LOG_ZERO);
	double * fvs = THDoubleTensor_data(fvsT);
	
	fvs[0] = aligned[0];
	fvs[1] = aligned[1];
	
	int lower_bound = -1, upper_bound = 2;
	
	double fvs_tmp, fvs_i1u, fvs_i1u1, fvs_i1u2;
	
	for(int i = 1; i < T; i++) {
		// adjust bounds, some positions would never been visited
		
		upper_bound += 2;
		if (upper_bound > L) {
			upper_bound = L;
		}
		
		lower_bound = L - 2 * (T - i);
		if (lower_bound < 0) {
			lower_bound = 0;
		}

		assert(lower_bound >= 0 && lower_bound < T * L);
		assert(upper_bound >= 0 && upper_bound < T * L);

		for (int u = lower_bound; u < upper_bound; u++) {
			double tmp = LOG_ZERO;
			
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

static THDoubleTensor * __get_backward_variable(THDoubleTensor * outputTable, THDoubleTensor * alignedTable, THDoubleTensor * targetT) {
	int T = outputTable->size[0];
	int L = targetT->size[0];
	
	double * aligned = THDoubleTensor_data(alignedTable);
	double * target = THDoubleTensor_data(targetT);
	
	THDoubleTensor * bvsT = THDoubleTensor_newWithSize2d(T, L);
	THDoubleStorage_fill(bvsT->storage, LOG_ZERO);
	double * bvs = THDoubleTensor_data(bvsT);
	
	assert(T * L >= 2);

	bvs[T * L - 1] = 0;
	bvs[T * L - 2] = 0;
	
	int lower_bound = -1, upper_bound = L - 3;
	
	double bvs_tmp, bvs_i1u, bvs_i1u1, bvs_i1u2;
	
	
	for(int i = T - 2; i >= 0; i--) {
		// adjust bounds, some positions would never been visited
		
		upper_bound -= 2;
		if (upper_bound < 0) {
			upper_bound = 0;
		}
		
		lower_bound = 2 * i + 1;
		if (lower_bound > L - 1) {
			lower_bound = L - 1;
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
			
			double tmp = LOG_ZERO;
			
			assert((i * L + u < T * L) && (i * L + u) >= 0);
			assert(((i + 1) * L + u) >= 0 && ((i + 1) * L + u) < T * L);
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
			
			if ((u < L - 1) && (i + 1) * L + u + 1 >= T * L) {
				perror("out of range\n");
			}
		}
		
	}
	return bvsT;
}


static THDoubleTensor * __get_grad(THDoubleTensor * fbT, THDoubleTensor * outputTable, THDoubleTensor * targetT, double pzx) {
	
	int T = fbT->size[0];
	int L = targetT->size[0];
	int class_num = outputTable->size[1];
	
	int pos;
	
	THDoubleTensor * gradT = THDoubleTensor_newWithSize2d(T, class_num);
	double * fb = THDoubleTensor_data(fbT);
	double * output = THDoubleTensor_data(outputTable);
	double * grad = THDoubleTensor_data(gradT);
	double * target = THDoubleTensor_data(targetT);
	
	double tmp_sum = 0, u = 0, tmp = 0;

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
	THDoubleTensor * output = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	THDoubleTensor * alignedTable = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	THDoubleTensor * target = (THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");
	
	
	
	THDoubleTensor * fvs = __get_forward_variable(output, \
													alignedTable, target);
	
												
	luaT_pushudata(L, fvs, "torch.DoubleTensor");
	
	return 1;											
}

static int ctc_get_backward_variable(lua_State * L) {
	THDoubleTensor * output = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	THDoubleTensor * alignedTable = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	THDoubleTensor * target = (THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");
	
	
	
	THDoubleTensor * bvs = __get_backward_variable(output, \
													alignedTable, target);
	
	/*
	double * data = THDoubleTensor_data(bvs);
	
	for (int i = 0; i < bvs->size[0]; i++) {
		for (int j = 0; j < bvs->size[1]; j++) {
			printf("%.4f\t", data[i * bvs->size[1] + j] == -1e10 ? 0 : data[i * bvs->size[1] + j]);
		}
		printf("\n");
	}
	*/
									
	luaT_pushudata(L, bvs, "torch.DoubleTensor");
	
	return 1;											
}



static int ctc_get_grad(lua_State * L) {
	THDoubleTensor * fb = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	THDoubleTensor * outputTable = (THDoubleTensor *)luaT_checkudata(L, 2, "torch.DoubleTensor");
	THDoubleTensor * target = (THDoubleTensor *)luaT_checkudata(L, 3, "torch.DoubleTensor");
	double pzx = luaL_checknumber(L, 4);
	
	THDoubleTensor * grad = __get_grad(fb, outputTable, target, pzx);
	
	luaT_pushudata(L, grad, "torch.DoubleTensor");
	
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
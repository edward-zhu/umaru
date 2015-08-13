extern "C" {
#include <luaT.h>
#include <TH/TH.h>
}

#include <cmath>
#include <iostream>
#include <vector>

static int ctc_print(lua_State * L)
{
	THDoubleTensor * input = (THDoubleTensor *)luaT_checkudata(L, 1, "torch.DoubleTensor");
	int h = input->size[0];
	int w = input->size[1];
	
	double * data = THDoubleTensor_data(input);
	
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			printf("%.4lf\t", data[i * w + j]);
		}
		printf("\n");
	}
	
	return 0;
}

static const struct luaL_reg ctc[] = {
	{"ctc_print", ctc_print},
	{NULL, NULL}
};

LUA_EXTERNC int luaopen_ctc(lua_State *L) {
	luaL_openlib(L, "ctc", ctc, 0);
	return 1;
}
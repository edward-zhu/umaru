CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenMP)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

foreach(target libctc normalizer)
	message(${target})
	add_library(${target} SHARED "${target}.cc")
	set_target_properties(${target}
		PROPERTIES PREFIX ""
		SUFFIX ".so"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
	)
	target_link_libraries(${target} TH luaT luajit)
endforeach(target)



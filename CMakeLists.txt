CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

foreach(target ctc normalizer)
	message(${target})
	add_library(${target} SHARED "${target}.cc")
	set_target_properties(${target}
		PROPERTIES PREFIX ""
		SUFFIX ".so"
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../"
	)
	target_link_libraries(${target} TH luaT luajit)
endforeach(target)



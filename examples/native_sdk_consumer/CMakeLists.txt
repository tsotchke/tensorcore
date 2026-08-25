cmake_minimum_required(VERSION 3.20)
project(tensorcore_native_sdk_consumer LANGUAGES C CXX)

find_package(tensorcore CONFIG REQUIRED)

add_executable(consumer_shared main.c)
target_link_libraries(consumer_shared PRIVATE tensorcore::tensorcore_shared)
if(COMMAND tensorcore_copy_metallib)
    tensorcore_copy_metallib(consumer_shared)
endif()

add_executable(consumer_static main.c)
target_link_libraries(consumer_static PRIVATE tensorcore::tensorcore)
if(COMMAND tensorcore_copy_metallib)
    tensorcore_copy_metallib(consumer_static)
endif()

add_executable(consumer_cxx main.cpp)
target_link_libraries(consumer_cxx PRIVATE tensorcore::tensorcore_shared)
if(COMMAND tensorcore_copy_metallib)
    tensorcore_copy_metallib(consumer_cxx)
endif()

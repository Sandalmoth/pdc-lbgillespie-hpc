cmake_minimum_required(VERSION 3.0.0)

add_executable(lbprocess main.cpp)
add_executable(nlb test2.cpp)
add_executable(dis testd.cpp)
add_executable(dmut testm.cpp)
add_executable(chng testc.cpp)
add_executable(two twofactor.cpp)

# Compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++17")

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -ggdb") # debug compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Werror")
endif()

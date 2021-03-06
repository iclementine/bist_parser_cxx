PROJECT(lstm-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-funroll-loops -Ofast -march=native")

if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
endif()
set(Boost_REALPATH ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

set(CMAKE_BUILD_TYPE "Release")
add_executable(archybrid-parser arc-hybrid.cc)
add_executable(test-corpus test_corpus.cc)
target_link_libraries(archybrid-parser ${Boost_LIBRARIES} rt dynet ${LIBS})

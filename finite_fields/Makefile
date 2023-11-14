NAME = main
SOURCES = utils.cpp

CXX = nvcc
CXXFLAGS = -std=c++17 -O3 -rdc=true -x cu $(SOURCES)

.DEFAULT_GOAL := $(NAME)

.PHONY: clean
clean:
	rm -rf $(NAME)

appname := main

sources := $(wildcard *.cpp)
objects := $(patsubst %.cpp, %.o, $(sources))
depends := $(patsubst %.cpp, %.d, $(sources))

CXX := g++
CXXFLAGS := -Wall -g -Wextra

all: $(appname)

$(appname): $(objects)
	$(CXX) $(CXXFLAGS) $^ -o $@

-include $(depends)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -fr *.o *.d $(appname)

run:
	./$(appname)

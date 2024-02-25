appname := main

sources := $(wildcard *.cpp)
objects := $(patsubst %.cpp, %.o, $(sources))
depends := $(patsubst %.cpp, %.d, $(sources))

CXX := g++
CXXFLAGS := -Wall -g

all: $(appname)

$(appname): $(objects)
	$(CXX) $(CXXFLAGS) $^ -o $@

-include $(depends)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -fr src/*.o tests/*.o tests/*.d src/*.d src/*.gcda src/main src/*.gcno src/*.h.gch src/**.egg-info pyllumina/__pycache__ $(appname) build dist *.so *.egg-info .eggs gmon.out gprof.png nnd.ltrans* nnd.wpa.gcno
	ctags -R .



run:
	./$(appname)

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
CPPFLAGS += -Isrc -MMD -MP

SRCS := $(shell find src -name '*.cpp')
OBJS := $(SRCS:src/%.cpp=build/%.o)

bitmoodmouth: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

build/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf build bitmoodmouth

.PHONY: clean

-include $(OBJS:.o=.d)

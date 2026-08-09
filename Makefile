CXX ?= c++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wno-unused-variable -mavx2
NORMAL_CXXFLAGS := $(filter-out -mavx2,$(CXXFLAGS))

TARGETS := FixEncoderLowLatency FixEncoder_Normal

.PHONY: all clean run run-normal compare

all: $(TARGETS)

FixEncoderLowLatency: FixEncoderLowLatency.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

FixEncoder_Normal: FixEncoder_Normal.cpp
	$(CXX) $(NORMAL_CXXFLAGS) $< -o $@

run: FixEncoderLowLatency
	./FixEncoderLowLatency

run-normal: FixEncoder_Normal
	./FixEncoder_Normal

compare: $(TARGETS)
	@echo Optimized encoder
	./FixEncoderLowLatency
	@echo Normal encoder
	./FixEncoder_Normal

clean:
	rm -f $(TARGETS)

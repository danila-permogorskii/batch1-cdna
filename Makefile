ARCH     ?= gfx942
ROCM     ?= /opt/rocm
HIPCXX   ?= $(ROCM)/lib/llvm/bin/amdclang++
CXXFLAGS  = -x hip -O3 -std=c++17 --offload-arch=$(ARCH) --rocm-path=$(ROCM) -I common
LDFLAGS   = -L$(ROCM)/lib -lamdhip64

BINS = 01-ceiling/ceiling 02-gemv/gemv 03-dlops/dlops 04-reduce/reduce 05-sync/sync

all: $(BINS)

$(BINS): %: %.hip
	$(HIPCXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

%.s: %.hip
	$(HIPCXX) $(CXXFLAGS) --offload-device-only -S -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BINS) */*.s

.PHONY: all clean
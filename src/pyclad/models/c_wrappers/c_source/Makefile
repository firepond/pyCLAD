## compiler - Use clang from Homebrew's llvm installation
CC=/opt/homebrew/opt/llvm/bin/clang

## Linker flags to find the OpenMP library
LDFLAGS = -L/opt/homebrew/opt/llvm/lib

# directories
ODIR=lib
#files
LOF_SRC = fogml_lof.c fogml_lof.h
LOF_OBJ = $(ODIR)/fogml_lof.so
# default target
all: $(LOF_OBJ)
# rule to build shared objects
$(ODIR)/%.so: %.c
		mkdir -p $(@D)
		$(CC) -Ofast -ffast-math  -funroll-loops -fPIC -mcpu=apple-m2 -fopenmp -arch arm64 -shared -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
		rm -rf $(ODIR)/*.so
## compiler
CC=gcc
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
		$(CC) -fPIC -shared -o $@ $<

.PHONY: clean
clean:
		rm -rf $(ODIR)/*.so
## compiler
CC=clang
# directories
ODIR=./obj
#files
SRC1 = test.c
OBJ1 = $(ODIR)/test.so
# default target
all: $(OBJ1)
# rule to build shared objects
$(ODIR)/%.so: %.c
		mkdir -p $(@D)
		$(CC) -fPIC -shared -o $@ $<

.PHONY: clean
clean:
		rm -rf $(ODIR)/*.so
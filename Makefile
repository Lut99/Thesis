GCC=gcc
GCC_ARGS=-std=c11 -O2 -Wall -Wextra

SRC=src
LIB=$(SRC)/lib
BIN=bin
OBJ=$(BIN)/obj
TST=tests
TST_BIN=$(BIN)/tests

INCLUDES=-I $(LIB)/include

.PHONY: default
default: all

$(OBJ)/%.o: $(LIB)/%.c
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $<

test_matrix: $(TST)/test_matrix.c $(OBJ)/Matrix.o
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/Matrix.o

tests: test_matrix
	$(info )
	$(info Running tests...)
	$(info )
	$(TST_BIN)/test_matrix.out

all: tests

clean:
	rm -f $(BIN)/*.out
	rm -f $(TST_BIN)/*.out
	rm -f $(OBJ)/*.o
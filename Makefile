GCC=gcc
GCC_ARGS=-std=c11 -O2 -Wall -Wextra
EXT_LIBS=-lm

ifdef DEBUG
GCC_ARGS+=-g
endif

ifdef PROFILE
GCC_ARGS+=-pg
endif

SRC=src
LIB=$(SRC)/lib
BIN=bin
OBJ=$(BIN)/obj
TST=tests
TST_BIN=$(BIN)/tests

INCLUDES=-I $(LIB)/include

NN_VERSION=$(OBJ)/NeuralNetwork.o

ifdef OPENMP
GCC_ARGS+=-fopenmp
NN_VERSION=$(OBJ)/NeuralNetwork_OpenMP_$(OPENMP).o
endif

.PHONY: default dirs plot
default: all

$(BIN):
	mkdir -p $@
$(OBJ):
	mkdir -p $@
$(TST_BIN):
	mkdir -p $@
dirs: $(BIN) $(OBJ) $(TST_BIN)

$(OBJ)/%.o: $(LIB)/%.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)
$(OBJ)/NeuralNetwork_%.o: $(LIB)/Optimisation/NeuralNetwork_%.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)

$(OBJ)/NeuralNetwork.a: $(NN_VERSION) $(OBJ)/Functions.o $(OBJ)/Array.o $(OBJ)/Matrix.o | dirs
	ar cr $@ $(NN_VERSION) $(OBJ)/Functions.o $(OBJ)/Array.o $(OBJ)/Matrix.o

$(BIN)/digits.out: $(SRC)/Digits.c $(OBJ)/NeuralNetwork.a | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ $< $(OBJ)/NeuralNetwork.a $(EXT_LIBS)

digits: $(BIN)/digits.out

test_matrix: $(TST)/test_matrix.c $(OBJ)/Matrix.o
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/Matrix.o $(EXT_LIBS)

test_nn: $(TST)/test_nn.c $(OBJ)/NeuralNetwork.a
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/NeuralNetwork.a $(EXT_LIBS)

test_array: $(TST)/test_array.c $(OBJ)/Array.o
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/Array.o $(EXT_LIBS)

tests: test_matrix test_nn test_array
	$(info )
	$(info Running tests...)
	$(info )
	$(TST_BIN)/test_matrix.out
	$(TST_BIN)/test_nn.out
	$(TST_BIN)/test_array.out

plot:
	gnuplot -e "set terminal png size 600,400; set output 'nn_costs.png'; set yrange[0:]; plot \"nn_costs.dat\""

all: digits tests plot

clean:
	rm -f $(BIN)/*.out
	rm -f $(TST_BIN)/*.out
	rm -f $(OBJ)/*.o
	rm -f $(OBJ)/*.a
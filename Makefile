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

ifdef VARIATION
ifneq ($(VARIATION), Sequential)
GCC_ARGS+=-fopenmp
endif
else
VARIATION=Sequential
endif

.PHONY: default dirs digits tests plot all
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

$(OBJ)/NeuralNetwork_%.o: $(LIB)/NeuralNetwork/NeuralNetwork_%.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)
$(OBJ)/Digits.o: $(SRC)/Digits.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)

$(OBJ)/Support.a: $(OBJ)/Functions.o $(OBJ)/Array.o $(OBJ)/Matrix.o | dirs
	ar cr $@ $(OBJ)/Functions.o $(OBJ)/Array.o $(OBJ)/Matrix.o

$(BIN)/digits_%.out: $(OBJ)/NeuralNetwork_%.o $(OBJ)/Digits.o $(OBJ)/Support.a | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ $< $(OBJ)/Digits.o $(OBJ)/Support.a $(EXT_LIBS)

$(TST_BIN)/playground.out: $(TST)/playground.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ $< $(EXT_LIBS)

digits: $(BIN)/digits_sequential.out

playground: $(TST_BIN)/playground.out

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

all: digits tests plot playground

clean:
	rm -f $(BIN)/*.out
	rm -f $(TST_BIN)/*.out
	rm -f $(OBJ)/*.o
	rm -f $(OBJ)/*.a
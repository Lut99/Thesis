GCC=gcc
GCC_ARGS=-std=c11 -O2 -Wall -Wextra

SRC=src
LIB=$(SRC)/lib
BIN=bin
OBJ=$(BIN)/obj
TST=tests
TST_BIN=$(BIN)/tests

INCLUDES=-I $(LIB)/include

.PHONY: default plot
default: all

$(OBJ)/%.o: $(LIB)/%.c
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< -lm

$(OBJ)/NeuralNetwork.a: $(OBJ)/NeuralNetwork.o $(OBJ)/Functions.o $(OBJ)/Matrix.o
	ar cr $@ $(OBJ)/NeuralNetwork.o $(OBJ)/Functions.o $(OBJ)/Matrix.o

$(BIN)/digits.out: $(SRC)/Digits.c $(OBJ)/NeuralNetwork.a
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ $< $(OBJ)/NeuralNetwork.a -lm

digits: $(BIN)/digits.out

test_matrix: $(TST)/test_matrix.c $(OBJ)/Matrix.o
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/Matrix.o -lm

test_nn: $(TST)/test_nn.c $(OBJ)/NeuralNetwork.a
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $(TST_BIN)/$@.out $< $(OBJ)/NeuralNetwork.a -lm

tests: test_matrix test_nn
	$(info )
	$(info Running tests...)
	$(info )
	$(TST_BIN)/test_matrix.out
	$(TST_BIN)/test_nn.out

plot:
	gnuplot -e "set terminal png size 600,400; set output 'nn_costs.png'; set yrange[0:]; plot \"nn_costs.dat\""

all: digits tests plot

clean:
	rm -f $(BIN)/*.out
	rm -f $(TST_BIN)/*.out
	rm -f $(OBJ)/*.o
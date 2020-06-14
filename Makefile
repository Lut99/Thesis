GCC=gcc
GCC_ARGS=-std=c11 -O2 -Wall -Wextra
NVCC=nvcc
NVCC_ARGS=-O2 --gpu-architecture=compute_75 --gpu-code=sm_75

EXT_LIBS=-lm

ifdef DEBUG
GCC_ARGS+=-g
NVCC_ARGS+=-g
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

INCLUDES=-I$(LIB)/include

ifdef BENCHMARK
GCC_ARGS+=-DBENCHMARK
NVCC_ARGS+=-DBENCHMARK
endif

ifdef VARIATION
ifneq ($(VARIATION), sequential)
GCC_ARGS+=-fopenmp
ifneq (,$(findstring omp_gpu,$(shell echo $(VARIATION) | tr A-Z a-z)))
# Set the compiler to the one in /var/scratch/tmuller
GCC=/var/scratch/tmuller/opt/offload/install/bin/gcc
# Add the offloading args
GCC_ARGS+= -foffload="-lm" -foffload=nvptx-none
endif
endif
else
VARIATION=sequential
endif

.PHONY: default dirs digits testdata tests plot all
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

$(OBJ)/Support.a: $(OBJ)/Array.o $(OBJ)/NeuralNetwork.o | dirs
	ar cr $@ $^

$(OBJ)/NeuralNetwork_%.o: $(LIB)/NeuralNetwork/NeuralNetwork_%.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)
$(OBJ)/NeuralNetwork_%.o: $(LIB)/NeuralNetwork/NeuralNetwork_%.cu | dirs
	$(NVCC) $(NVCC_ARGS) $(INCLUDES) -o $@ --device-c $< $(EXT_LIBS)
$(OBJ)/Digits.o: $(SRC)/Digits.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)
$(OBJ)/TestData.o: $(SRC)/TestData.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $< $(EXT_LIBS)

$(BIN)/digits.out: $(OBJ)/NeuralNetwork_${VARIATION}.o $(OBJ)/Digits.o $(OBJ)/Support.a | dirs
	$(NVCC) $(NVCC_ARGS) $(INCLUDES) -o $@ $^ $(EXT_LIBS)

$(BIN)/testdata.out: $(OBJ)/TestData.o $(OBJ)/NeuralNetwork_${VARIATION}.o $(OBJ)/Support.a | dirs
	$(NVCC) $(NVCC_ARGS) $(INCLUDES) -o $@ $^ $(EXT_LIBS)

$(TST_BIN)/playground.out: $(TST)/playground.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ $< $(EXT_LIBS)

digits: $(BIN)/digits.out

digits_cuda: $(BIN)/digits_cuda.out
	mv $(BIN)/digits_cuda.out $(BIN)/digits.out

testdata: $(BIN)/testdata.out

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

$(OBJ)/debug_cuda.o: $(TST)/debug_cuda.c | dirs
	$(GCC) $(GCC_ARGS) $(INCLUDES) -o $@ -c $^ $(EXT_LIBS)

$(TST_BIN)/debug_cuda.out: $(OBJ)/debug_cuda.o $(OBJ)/Support.a $(OBJ)/NeuralNetwork_CUDA_GPU1.o | dirs
	$(NVCC) $(NVCC_ARGS) $(INCLUDES) -o $@ $^ $(EXT_LIBS)

debug_cuda: $(TST_BIN)/debug_cuda.out

plot:
	gnuplot -e "set terminal png size 600,400; set output 'nn_costs.png'; set yrange[0:]; plot \"nn_costs.dat\""

all: digits testdata tests plot playground

clean:
	rm -f $(BIN)/*.out
	rm -f $(TST_BIN)/*.out
	rm -f $(OBJ)/*.o
	rm -f $(OBJ)/*.a
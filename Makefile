NVCC=nvcc
NVCCFLAGS=-std=c++17
NVCCFLAGS+=-gencode arch=compute_60,code=sm_60
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-I./include -I./src/cutf/include

TARGET=lib/libcu_exp_statistics.a

$(TARGET):obj/main.o obj/main.dlink.o
	mkdir -p lib
	$(NVCC) $< -o $@ $(NVCCFLAGS) -lib

obj/main.o:src/main.cu
	mkdir -p obj
	$(NVCC) $< -o $@ $(NVCCFLAGS) -dc

obj/main.dlink.o:obj/main.o
	mkdir -p obj
	$(NVCC) $< -o $@ $(NVCCFLAGS) -dlink

test: test/main.cu $(TARGET)
	$(NVCC) $+ -o $@.out $(NVCCFLAGS)
  
clean:
	rm -rf $(TARGET) obj

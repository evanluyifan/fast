all: nms_test 

nms_test: nms_main.o dlink_nms_kernel.o
	g++ nms_main.o dlink_nms_kernel.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -o nms_test

nms_main.o: dlink_nms_kernel.o
	nvcc -c -arch=sm_52 main.cpp -o nms_main.o 

dlink_nms_kernel.o:
	nvcc -c -arch=sm_52 -rdc=false nms_kernel.cu -o dlink_nms_kernel.o

clean: 
	rm -rf *.o nms_test 

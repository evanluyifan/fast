#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <cstring>
#include <fstream>

#include <cuda_runtime.h>

#include "nms_kernel.h"

#define BOX_SIZE 4

#define CUDA_CHECK(condition) \
    do { \
        cudaError_t error = condition; \
        if (error != cudaSuccess) { \
            std::cout << cudaGetErrorString(error) << std::endl; \
        } \
    } while (0)



int main () {
#include "testdata.h"

    const float overlap_thresh = 0.3;
    const float boxsize_thresh = 0.1;

    const int batch = BATCH;
    const int batch_size = TEST_BOXES_NUM_PER_BATCH;
    const int output_boxes_num = 300;

    float output_boxes[BATCH * output_boxes_num * BOX_SIZE];

    std::memset(output_boxes, -1, BATCH * output_boxes_num * BOX_SIZE * sizeof(float));
    
    float* boxes_dev = NULL;
    size_t input_array_size = batch * batch_size * BOX_SIZE * sizeof(float);

    float* boxes_output_dev = NULL;
    size_t output_array_size = batch * output_boxes_num * BOX_SIZE * sizeof(float);

    CUDA_CHECK(cudaMalloc(&boxes_dev, input_array_size));
    CUDA_CHECK(cudaMalloc(&boxes_output_dev, output_array_size));

    /*** test ***/
    CUDA_CHECK(cudaMemset(boxes_output_dev, 0, output_array_size));

    CUDA_CHECK(cudaMemcpy(boxes_dev,
                    test_boxes,
                    input_array_size,
                    cudaMemcpyHostToDevice));

    //struct timeval tm_start, tm_stop;
    //gettimeofday(&tm_start, NULL);

    /*** measure kernel time ***/
    float time;
    cudaEvent_t e_start;
    cudaEvent_t e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start, 0);
    /*** measure kernel time ***/

    nms(boxes_dev,
                batch, 
                batch_size,
                output_boxes_num,
                overlap_thresh,
                boxsize_thresh,
                boxes_output_dev);

    /*** measure kernel time ***/
    cudaEventRecord(e_stop, 0);
    cudaEventSynchronize(e_start);
    cudaEventSynchronize(e_stop);
    cudaEventElapsedTime(&time, e_start, e_stop);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    printf(" | nms kernel time: %f ms\n", time);
    /*** measure kernel time ***/

    //gettimeofday(&tm_stop, NULL);

    CUDA_CHECK(cudaMemcpy(output_boxes,
                    boxes_output_dev,
                    output_array_size,
                    cudaMemcpyDeviceToHost));

    //double t_elapse = (1000000.0 * (tm_stop.tv_sec - tm_start.tv_sec) + (tm_stop.tv_usec - tm_start.tv_usec)) / 1000;
    //printf(" | time for nms: %3.1f msec\n", t_elapse);

    /*** print data ***/
    std::ofstream out_file("_nms_test_result.txt");
    if (out_file.is_open()) {
        for (int i = 0; i < output_boxes_num * BATCH; i++) {
            out_file << "|box| ";
            out_file << output_boxes[i * BOX_SIZE + 0] << ", ";
            out_file << output_boxes[i * BOX_SIZE + 1] << ", ";
            out_file << output_boxes[i * BOX_SIZE + 2] << ", ";
            out_file << output_boxes[i * BOX_SIZE + 3] << ",\n";
        }
        //out_file << " | time for nms: "<< t_elapse <<" msec\n";
        out_file << " | nms kernel time: " << time << " msec\n";
        out_file.close();

        printf(" | Results are printed out >> _nms_test_result.txt\n");
    }
    else {
        printf("WARNING: failed to print out results\n");
    }
    /*** print data ***/

    return 0;
}


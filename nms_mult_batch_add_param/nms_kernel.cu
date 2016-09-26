#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#include "nms_kernel.h"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define BOX_SIZE 4
#define BLOCK_SIZE 512

__forceinline__ __device__ void loadstore(float * dst, float const * const src) {
    #pragma unroll
    for (int k = 0; k < BOX_SIZE; k++) {
       dst[k] = src[k];
    }
}

__forceinline__ __device__ float getIoU(float const * const a, float const * const b) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

__forceinline__ __device__ bool checkBoxSize(float const * const box, const float boxsize_thresh) {
    float w = box[2] - box[0];
    float h = box[3] - box[1];
    return (min(w, h) > boxsize_thresh ? true : false);
}

__forceinline__ __device__ void setBoxesEmpty(float * boxes, const int start, const int end) {
    // start and end are the index of a box, and box have size
    int box_idx = threadIdx.x + start;
    int count = DIVUP((end - start), BLOCK_SIZE);
    #pragma unroll
    for (int i = 0; i < count; i++) {
        if (box_idx < end) {
            #pragma unroll
            for (int k = 0; k < BOX_SIZE; k++) {
                boxes[box_idx * BOX_SIZE + k] = 0.0f;
            }
        }
        box_idx += BLOCK_SIZE;
    }
}

__global__ 
__launch_bounds__(BLOCK_SIZE)
void nms_kernel(float *input_boxes,
            const int input_boxes_per_batch, 
            const int output_boxes_per_batch, 
            const float overlap_thresh, 
            const float boxsize_thresh,
            float *output_boxes) {

    extern __shared__ bool kept_boxes[];

    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    int batch_offset_in = batch_idx * input_boxes_per_batch;
    int batch_offset_out = batch_idx * output_boxes_per_batch;

    int max_box_idx = batch_offset_in + input_boxes_per_batch;

    int bpt = DIVUP((input_boxes_per_batch - thread_idx), BLOCK_SIZE); // bpt: num boxes per thread

    int cur_box_idx = batch_offset_in + thread_idx;
    float cur_box[BOX_SIZE];

    /*** check the box size and initialize kept_boxes ***/
    #pragma unroll
    for (int i = 0; i < bpt; i++) {
        loadstore(cur_box, input_boxes + cur_box_idx * BOX_SIZE);
        if (checkBoxSize(cur_box, boxsize_thresh)) {
            kept_boxes[cur_box_idx - batch_offset_in] = true;
        }
        else {
            kept_boxes[cur_box_idx - batch_offset_in] = false;
        }
        cur_box_idx += BLOCK_SIZE;
    }

    int ref_box_idx = 0 + batch_offset_in;

    int kept = 0; // record the number of boxes kept

    while (!kept_boxes[ref_box_idx - batch_offset_in] && ref_box_idx < max_box_idx) {
        ref_box_idx++;
    }
    // the first box may be removed for size too small

    /*** remove the overlaped boxes ***/
    while ((kept < output_boxes_per_batch) && (ref_box_idx < max_box_idx)) {
        float ref_box[BOX_SIZE];
        loadstore(ref_box, input_boxes + ref_box_idx * BOX_SIZE);

        cur_box_idx = batch_offset_in + thread_idx;

        #pragma unroll
        for (int i = 0; i < bpt; i++) {
            loadstore(cur_box, input_boxes + cur_box_idx * BOX_SIZE);
            if (cur_box_idx > ref_box_idx) {
                if (kept_boxes[cur_box_idx - batch_offset_in]) {
                    if (getIoU(ref_box, cur_box) > overlap_thresh) {
                        kept_boxes[cur_box_idx - batch_offset_in] = false;
                    }
                }
            }
            else if (cur_box_idx == ref_box_idx) {
                loadstore(output_boxes + (batch_offset_out + kept) * BOX_SIZE, ref_box);
            }
            cur_box_idx += BLOCK_SIZE;
        }

        __syncthreads();

        do {
            ref_box_idx++;
        } while (!kept_boxes[ref_box_idx - batch_offset_in] && (ref_box_idx < max_box_idx));

        kept++;
    }

    /*** if kept boxes < request output boxes, set left output buffer 0 ***/
    if (kept < output_boxes_per_batch) {
        setBoxesEmpty(output_boxes + batch_offset_out * BOX_SIZE,
                    batch_offset_out + kept,
                    batch_offset_out + output_boxes_per_batch);
    }
}

void nms(float* boxes_input, 
            const int batches, 
            const int input_boxes_per_batch, 
            const int output_boxes_per_batch, 
            const float overlap_thresh, 
            const float boxsize_thresh, 
            float* boxes_output) {

    nms_kernel <<< batches, BLOCK_SIZE, input_boxes_per_batch>>> 
        (boxes_input,
         input_boxes_per_batch, 
         output_boxes_per_batch, 
         overlap_thresh,
         boxsize_thresh,
         boxes_output);
}

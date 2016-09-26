#ifndef __NMS_KERNEL_HEADER_INTERNAL__
#define __NMS_KERNEL_HEADER_INTERNAL__
void nms(float* boxes_host, 
            const int batch, 
            const int input_boxes_per_batch, 
            const int output_boxes_per_batch, 
            const float overlap_thresh, 
            const float boxsize_thresh, 
            float* boxes_output);

__global__
void nms_kernel(float *dev_input_boxes,
            const int input_boxes_per_batch, 
            const int output_boxes_per_batch, 
            const float overlap_thresh, 
            const float boxsize_thresh,
            float *dev_output_boxes);
#endif

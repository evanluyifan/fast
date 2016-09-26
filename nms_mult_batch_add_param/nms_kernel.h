#ifndef __NMS_KERNEL_HEADER__
#define __NMS_KERNEL_HEADER__
void nms(float* boxes_host, 
            const int batch, 
            const int input_boxes_per_batch, 
            const int output_boxes_per_batch, 
            const float overlap_thresh, 
            const float boxsize_thresh, 
            float* boxes_output);
#endif

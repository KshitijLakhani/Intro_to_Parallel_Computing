/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "utils.h"
#include <stdio.h>   /* This is for printf*/
#include <limits>

__global__ void shmem_min_max_reduce_kernel(float * d_min, const float * d_in, const int thread_count)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t,
    // shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    if (tid < thread_count/2){
	sdata[tid] = d_in[myId];
    	__syncthreads();      
	}
	else{
	sdata[tid] = d_in
      // make sure entire block is loaded!
    	sdata[tid+thread_count] = d_in[myId];
   	 __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
        sdata[tid] = min(sdata[tid],sdata[tid+s]);
	sdata[tid + thread_count] = max(sdata[tid + thread_count],sdata[tid + thread_count + s]);
        /*sdata[tid] += sdata[tid + s];*/
        }
        __syncthreads();        // make sure all adds at one stage are
//done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_min[blockIdx.x] = sdata[0];
    }

    
}




__global__ void shmem_max_reduce_kernel(float * d_max, const float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t,
    // shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
        sdata[tid] = max(sdata[tid],sdata[tid+s]);
        /*sdata[tid] += sdata[tid + s];*/
        }
        __syncthreads();        // make sure all adds at one stage are
//done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_max[blockIdx.x] = sdata[0];
    }
}




void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

/*This was just to print the d_logLuminance array*/
/*
float* h_test;
h_test = new float[numRows*numCols];

cudaMemcpy(h_test, d_logLuminance, sizeof(float) * (numRows*numCols), cudaMemcpyDeviceToHost);

for (unsigned int i=0;i<numRows*numCols;i++){
printf("%f ",h_test[i]);
}
*/

const size_t numPixels = numRows * numCols;
const int THREAD_COUNT = 1024;
const int BLOCK_COUNT = numPixels/THREAD_COUNT;

/********Serial calculation of min and max values************/
/*Creating temporary array on host*/
float temp_arr[numPixels];
/*Transfering logLuminance input vector from device to host*/
checkCudaErrors(cudaMemcpy(temp_arr, d_logLuminance, sizeof(float) * numPixels, cudaMemcpyDeviceToHost));

float logLumMin = temp_arr[0];
float logLumMax = temp_arr[0];

  for (size_t i = 1; i < numPixels; ++i) {
    logLumMin = std::min(temp_arr[i], logLumMin);
    logLumMax = std::max(temp_arr[i], logLumMax);
  }
/*Print the serial values*/
printf("Serial min = %f",logLumMin);
printf("Serial max = %f",logLumMax);
/*delete [] temp_arr;*/     /*To delete temporary array*/

/************End of serial calculation*******************************/


/************Parallel Calculation of min and max******************/
/*Let us define 2 different device variables to store the max and min output*/
float* d_min_op;
float* d_max_op;

checkCudaErrors(cudaMalloc(&d_min_op,sizeof(float) * (BLOCK_COUNT + 32)));   // 96 + 32 = 128
checkCudaErrors(cudaMalloc(&d_max_op,sizeof(float) * (BLOCK_COUNT + 32)));   

/*Kernel calls*/
/*padding array*/
float h_min_pad[32];
float h_max_pad[32];

for(int k=0;k<32;k++) {
h_min_pad[k]=std::numeric_limits<float>::max();
h_max_pad[k]=std::numeric_limits<float>::min();
}

checkCudaErrors(cudaMemcpy(&d_min_op[96],h_min_pad,sizeof(float)*32,cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(&d_max_op[96],h_max_pad,sizeof(float)*32,cudaMemcpyHostToDevice));

shmem_min_reduce_kernel<<<BLOCK_COUNT,THREAD_COUNT,sizeof(float)*THREAD_COUNT>>>(d_min_op, d_logLuminance);
shmem_min_reduce_kernel<<<1,(BLOCK_COUNT+32),sizeof(float)*(BLOCK_COUNT+32)>>>(d_min_op, d_min_op);
checkCudaErrors(cudaMemcpy(&min_logLum, d_min_op, sizeof(float), cudaMemcpyDeviceToHost));

shmem_max_reduce_kernel<<<BLOCK_COUNT,THREAD_COUNT,sizeof(float)*THREAD_COUNT>>>(d_max_op, d_logLuminance);
shmem_max_reduce_kernel<<<1,(BLOCK_COUNT+32),sizeof(float)*(BLOCK_COUNT+32)>>>(d_max_op, d_max_op);
checkCudaErrors(cudaMemcpy(&max_logLum, d_max_op, sizeof(float), cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(d_min_op));
checkCudaErrors(cudaFree(d_max_op));

printf("Parallel Min is %f",min_logLum);
printf("Parallel Max is %f",max_logLum);


  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


}

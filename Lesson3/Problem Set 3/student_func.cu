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
#include <math.h>
__global__ void shmem_min_reduce_kernel(float * d_min, const float * d_in)
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
        sdata[tid] = min(sdata[tid],sdata[tid+s]);
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
 //   __syncthreads();            // make sure entire block is loaded!

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


__global__ void hist_bin_kernel(const float* const d_lum,float lumMin,const size_t numBins,const float range,float* d_bin_arr,const size_t  numPixels)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //int tid  = threadIdx.x;
	if(myId > numPixels){
	return;
	}

	int d_bin = (int)round(((d_lum[myId] - lumMin)/range)*(numBins-1));
	/*d_bin_arr[bin]=bin_arr[bin] + 1;*/
	atomicAdd(&d_bin_arr[d_bin], 1);
}



__global__ void prescan(unsigned int * const g_odata, float *g_idata, const size_t n)
{
 extern __shared__ float temp[];// allocated on invocation
 int tid = threadIdx.x;
 int offset = 1;
 temp[2*tid] = g_idata[2*tid]; // load input into shared memory
 temp[2*tid+1] = g_idata[2*tid+1];
// __synthreads();

 for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
 {
 __syncthreads();
 if (tid < d)
 {
int ai = offset*(2*tid+1)-1;
 int bi = offset*(2*tid+2)-1;
 temp[bi] += temp[ai];
 }
 offset *= 2;
 }
 if (tid == 0) { temp[n - 1] = 0; } // clear the last element
 for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
 {
 offset >>= 1;
 __syncthreads();
 if (tid < d)
 {
int ai = offset*(2*tid+1)-1;
 int bi = offset*(2*tid+2)-1;
 float t = temp[ai];
 temp[ai] = temp[bi];
 temp[bi] += t;
 }
 }
 __syncthreads();
 g_odata[2*tid] = static_cast<unsigned int>(temp[2*tid]); // write results to device memory
 g_odata[2*tid+1] = static_cast<unsigned int>(temp[2*tid+1]);
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
printf("Serial min = %f \n",logLumMin);
printf("Serial max = %f \n",logLumMax);
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

printf("Parallel Min is %f \n",min_logLum);
printf("Parallel Max is %f \n",max_logLum);

float lumRange = max_logLum - min_logLum;
printf("lumRange =%f \n",lumRange);


/********************Serial Histo***************/
/*Serial bin array*/
float bin_arr[numBins];
int bin;

for(unsigned int j=0;j<numBins;j++){
bin_arr[j]=0;
}

for(unsigned int j=0;j<numPixels;j++){
bin = (int)round(((temp_arr[j] - min_logLum)/lumRange)*(numBins-1));
bin_arr[bin]=bin_arr[bin] + 1;
} 

printf("For serial histogram \n bin0=%f bin1=%f bin2=%f bin3=%f bin4=%f bin100=%f bin567=%f bin1000=%f bin1023=%f \n",bin_arr[0],bin_arr[1],bin_arr[2],bin_arr[3],bin_arr[4],bin_arr[100],bin_arr[567],bin_arr[1000],bin_arr[1023]); 

float sum_serial=0;
/*Printing sum of all bin values*/
for(unsigned int j=0;j<numBins;j++){
sum_serial=sum_serial+bin_arr[j];
}

printf("Sum_serial=%f \n",sum_serial);
/*****************End of serial histo**********************/

/**************Parallel Histo*******************************/
/*Allocate memory on device to store d_bin_arr i.e. histogram*/
float* d_bin_arr;
checkCudaErrors(cudaMalloc(&d_bin_arr,numBins*sizeof(float)));
checkCudaErrors(cudaMemset(d_bin_arr,0,numBins*sizeof(float)));

/*Kernel call*/
hist_bin_kernel<<<BLOCK_COUNT,THREAD_COUNT>>>(d_logLuminance,min_logLum,numBins,lumRange,d_bin_arr,numPixels);

/*Temp array to copy d_bin_arr contents to host and print*/
float h_bin_arr[1024];
checkCudaErrors(cudaMemcpy(h_bin_arr,d_bin_arr,numBins*sizeof(float), cudaMemcpyDeviceToHost));

printf("For parallel histogram \n bin0=%f bin1=%f bin2=%f bin3=%f bin4=%f bin100=%f bin567=%f bin1000=%f bin1023=%f \n",h_bin_arr[0],h_bin_arr[1],h_bin_arr[2],h_bin_arr[3],h_bin_arr[4],h_bin_arr[100],h_bin_arr[567],h_bin_arr[1000],h_bin_arr[1023]);

float sum_parallel=0;
/*Printing sum of all bin values*/
for(unsigned int j=0;j<numBins;j++){
sum_parallel=sum_parallel + h_bin_arr[j];
}

printf("Sum_parallel=%f \n",sum_parallel);

/*for(unsigned int l=0;l<numBins;l++){
printf("h_bin_arr[%d]=%f \t",l,h_bin_arr[l]);
}*/

/*********Comparison function for serial and parallel ********************/
float count=0;
for(unsigned int l=0;l<numBins;l++){
	if(h_bin_arr[l] - bin_arr[l] != 0){
	count++;
}
}
printf("\n \n count = %f",count);

/*****************Scan Serial*****************/
 int  h_cdf_ser[1024];
h_cdf_ser[0]=0;
  for (size_t i = 1; i < numBins; ++i) {
    h_cdf_ser[i] = h_cdf_ser[i - 1] + h_bin_arr[i - 1];
  }

/**********End***************/

/*********************Scan Paralle****************/

prescan<<<1,(numBins/2),numBins*sizeof(float)>>>(d_cdf,d_bin_arr,numBins);

/*Temp array to copy d_cdf contents to host and print*/
unsigned int h_cdf[1024];
checkCudaErrors(cudaMemcpy(h_cdf,d_cdf,numBins*sizeof(unsigned int),cudaMemcpyDeviceToHost));

/*************ENd*******************/
float counter=1;
for(unsigned int z=0;z<numBins;z++){
        if((h_cdf[z] - h_cdf_ser[z]) != 0){
        counter++;
}
}
printf("\n \n counter = %f",counter);

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

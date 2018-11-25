/* Howto run ?
If executable name is 'reduce' then type the following in command terminal
./reduce 0 for global reduce
./reduce for shared memory reduce
 */

/* For global access, reduce run time - 0.911 ms approx
For shared memory access, reduce run time - 0.771 ms approx
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
/*
cuda.h -  defines the public host functions and types for the CUDA driver API.
cuda_runtime_api.h - defines the public host functions and types for the CUDA
runtime API
cuda_runtime.h - defines everything cuda_runtime_api.h does, as well as built-in
type definitions and function overlays for the CUDA language extensions and
device intrinsic functions.*/

/*All info about the .h files is from stach overflow*/

/*If you were writing host code to be compiled with the host compiler which
 * includes API calls, you would include either cuda.h or cuda_runtime_api.h.
 * If you needed other CUDA language built-ins, like types, and were using the
 * runtime API and compiling with the host compiler, you would include
 * cuda_runtime.h
*/

__global__ void global_reduce_kernel(float * d_out, float * d_in){
   int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();        // make sure all adds at one stage are
//done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
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
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are
//done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(float * d_out, float * d_intermediate, float * d_in, int size, bool usesSharedMemory)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads *sizeof(float)>>>(d_intermediate, d_in);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
    if (usesSharedMemory)
    {
      //  shmem_reduce_kernel<<<blocks, threads, threads *sizeof(float)>>>(d_out, d_intermediate);
    	global_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate);
    }
}

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
/*cudaGetDeviceCount ( int* count )

PARAMETERS
count

RETURNS
cudaSuccess, cudaErrorNoDevice, cudaErrorInsufficientDriver

DESCRIPTION
Returns in *count the number of devices with compute capability greater or equal to 2.0 that are available for execution. If there is no such device then
cudaGetDeviceCount() will return cudaErrorNoDevice. If no driver can be loaded to determine if any such devices exist then cudaGetDeviceCount() will return
cudaErrorInsufficientDriver.*/


    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
/*cudaError_t cudaSetDevice ( int  device )
PARAMETERS
device
- Device on which the active host thread should execute the device code.

RETURNS
cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceAlreadyInUse

DESCRIPTION
Sets device as the current device for the calling host thread. Valid device id's are 0 to (cudaGetDeviceCount() - 1).

Any device memory subsequently allocated from this host thread using cudaMalloc(), cudaMallocPitch() or cudaMallocArray() will be physically resident on
device. Any host memory allocated from this host thread using cudaMallocHost() or cudaHostAlloc() or cudaHostRegister() will have its lifetime associated with
device. Any streams or events created from this host thread will be associated with device. Any kernels launched from this host thread using the <<<>>>
operator or cudaLaunchKernel() will be executed on device.

This call may be made from any host thread, to any device, and at any time. This function will do no synchronization with the previous or new device, and
should be considered a very low overhead call.
*/

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
   /* for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }*/

//Additional part for input array on host - used for teesting correctness added by - Kshitij
	for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate non random float
        h_in[i] = 1.0f;
        sum += h_in[i];
    }



    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); //overallocated
    cudaMalloc((void **) &d_out, sizeof(float));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
	//whichKernel = argv[1]; -  incorrect. Explanation below.    
}

/*atoi example in this directory only
Run and see for understanding it 
Hence, when running reduce.cu we use ./reduce 0 or ./reduce 1
*/  
/*Why use the atoi ?
    When we receive argumnets from command line and stored in argv, they are stored as strings. Basically, argv is an array of strings.
	Now, while checking for shared memory or global memory usage, we use switch vase statements with number (int) - whichKernel.
	So,whichKernel = argv[1] won't work here as RHS is a string and LHS is an int.
	We need whichKernel to be int as used in the switch case statements.
	Hence, convert the received number - 0 or 1 which is stored as a string to an int and then use it for comparison ini switch case.
*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
    switch(whichKernel) {
    case 0:
        printf("Running global reduce\n");
        cudaEventRecord(start, 0);
        /*for (int i = 0; i < 100; i++)
        {
           reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
        }*/

	 //Addition by Kshitij - single call instead of 100 calls of the kernel
        reduce(d_out, d_intermediate,d_in,ARRAY_SIZE,false);
	cudaEventRecord(stop, 0);
        
	
	break;
    case 1:
        printf("Running reduce with shared mem\n");
        cudaEventRecord(start, 0);
        /*for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
        }*/
	reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
        cudaEventRecord(stop, 0);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials

    // copy back the sum from GPU
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("average time elapsed: %f\n", elapsedTime);
	
//Additional part - Kshitij
printf("Reduce answer :%f",h_out);
printf("Sum answer on host :%f",sum);



    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
        
    return 0;
}

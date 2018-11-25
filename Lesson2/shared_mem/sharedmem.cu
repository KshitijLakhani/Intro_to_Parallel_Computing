//When wold using shared memory make sense ?
//So, let's say you have a big array in the host memory and you transfer it to
//GPU memory and the task is to square each element of this array - This won't
//be a very good usage of __shared__ memory as you would first have to load
//from global to shared memory and then from shared memory to the thread
//memory for usage and there is no data reuse, so not a good example.Instead
//if you use normally, i.e. take the data from global to the thread memory
//then only 1 read so using shared memory here may make it slower on the
//contrary. 
//However, let's say, each output element of the array is such that it
//equals the corresponding value,squared, plus the average of both it's left
//and right neighbours, squared. 
//So for element arr[i] it would be arr[i]^2 + (arr[i-1]+arr[i+1]/2)^2. Here
//there is element reuse as is evident and hence shared memory makes sense to
//speed it up. 


#include <stdio.h>

//Kernel 
__global__ void compute(float *data)  //pointer to data in global memory which
//is later shifted to shared memory
{int tid = threadIdx.x;
__shared__ float arr[1024];
float temp;

//Load thread's data element in the shared memory
arr[tid] = data[tid];

//Now, before we startthe computation you want all the elements to be loaded
//in the shared memory. If a computation depends on an element in shared
//memory and if it's vlaues is not that from data[] but some earlier
//uninitialised vlaue then wrong computation, so use synchronization
__syncthreads();

temp = (arr[tid>0 ? tid-1 : 1023] + arr[tid<0 ? tid+1 : 0])*0.5f; //average
//calculation
//No need for synchronization now as the read from arr[] is not a problem and
//the write to temp is not a problem either as temp has been calculated in the
//earlier statement and then it is being altered, not depending on any other
//elements / threads

//temp = (data[tid>0 ? tid-1 :1023] + data[tid<0 ? tid+1 : 0])*0.5f;
//The statement above is used if no shared

temp = temp*temp + arr[tid]*arr[tid];
//The statement below is used if no shared
//temp = temp*temp + data[tid]*data[tid];
//Write back result to global memory straightaway. No need to write in shared
//memory and then update global as unneccessary extra step.

data[tid] = temp;
}

int main(int argc, char ** argv)
{
float data_h[1024];
float data_out_h[1024];
float *data_d;
int i;
//populate host
for(i=0;i<1023;i++){
data_h[i] = 2.0f;
}

//Device memory allocate
cudaMalloc((void **)&data_d,1024*sizeof(float));

//Transfer from host to device
cudaMemcpy(data_d,data_h,1024*sizeof(float),cudaMemcpyHostToDevice);

//Kernel call
compute<<<1,1024>>>(data_d);

//Copy backto host from device
cudaMemcpy(data_out_h,data_d,1024*sizeof(float),cudaMemcpyDeviceToHost);

//Print out results
for(i=0;i<1023;i++){
printf("%f ",data_out_h[i]);
}

//free memory
cudaFree(data_d);

return 0;
}

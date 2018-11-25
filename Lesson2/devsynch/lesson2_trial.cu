//This code is very similar to the one in the lectures and the one written in the notebook for further understanding.

#include <stdio.h>

__global__ void print_kernel(){
printf("Hello from block %d, thread %d \n",blockIdx.x,threadIdx.x);
}


__global__ void print2_kernel(){
printf("Threadidx.x=%d\n",threadIdx.x);
}

int main(){
print_kernel<<<10,10>>>();
print2_kernel<<<10,10>>>();
//What is the purpose of the cudaDeviceSynchronize?
//Seems like that if you use printf in the kernel, which we do end up doing, then what essentially happens is that all the outputs of the printf get stored in
//a circular buffer (usually approx 1MB size. This size can be changed using cudaDeviceSetLimit). So to flush these values out of the buffer, we have to use
//cudaDeviceSynchronize. When I say flush the values from the buffer, I mean that up until then the values only get stored in the buffer and not
//pushed/flushed out for printing on the screen. The cudaDeviceSynchronize ensures that this happens. If you remove cudaDeviceSynchronize then all values will
//remain in buffer but none of them would be printed/pushed/flushed out
cudaDeviceSynchronize();
}


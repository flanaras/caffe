#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <caffe/util/device_alternate.hpp>

// TODO: check offloading communication control logic in gpu paper for synchronisation techniques.
// see cuStream{Write,Wait}Value32

//typedef enum {none, waiting, requested, served} status;
enum status {none, waiting, requested, served, used};

struct data_exchange {
    //TODO: volatile variables
    status status = status::none;
    int requested_size = 0;
    int actual_size = 0;
    void* ptr = nullptr;
};

__host__ __device__ void blockUntil(data_exchange* dataExchange, status until) {
    while (true) {
        //TODO: read, with no race condition
        //TODO: CAS
        if (dataExchange->status == until) {
            return;
        }
    }
}

//from host to have a way to invoke data serving
__global__ void waitForDataRequest(data_exchange* dataExchange) {
    //TODO: CAS
    dataExchange->status = status::waiting;
    blockUntil(dataExchange, requested);
}

//from host when data are served to the GPU
__global__ void dataResponse(data_exchange* requestMount) {
    //blockUntil(requestMount, status::waiting);
    //TODO: copy things

    requestMount->status = status::served;
    blockUntil(requestMount, status::used);
    return;
}

__host__ __device__ void requestData(data_exchange* requestMount) {
    if (threadIdx.x == 0) {
        //TODO: write, with no race condition cas
        requestMount->status = status::requested;
    }
}

__global__ void data_layer_gpu(data_exchange* requestMount) {
    //TODO: How many things to transfer
    //TODO: Can assume count == blockDim.x
    atomicAdd(&requestMount->requested_size, 1);
    __syncthreads();

    requestData(requestMount);
    __syncthreads();

    //TODO: add image on the ith position in the blob
    //TODO: pass on the blob / return?
}

void thread2(data_exchange* cudaPrototype) {
    //execute listener
    //1 per block
    waitForDataRequest<<<1, 1>>>(cudaPrototype);

    //TODO: copy things to gpu
    dataResponse<<<1, 1>>>(cudaPrototype);
}

int main() {
    //do my foo bar
    data_exchange* prototype = nullptr;
    data_exchange* cudaPrototype = nullptr;
    prototype = (data_exchange*) malloc(sizeof(data_exchange));
    CUDA_CHECK(cudaMalloc(&cudaPrototype, sizeof(data_exchange)));

    prototype->requested_size = 0;
    prototype->actual_size = 0;
    prototype->status = status::none;
    prototype->ptr = nullptr;

    //cpy to cuda
    cudaMemcpy(cudaPrototype, prototype, sizeof(prototype), cudaMemcpyDefault);

    thread2(cudaPrototype);
    data_layer_gpu<<<1, 32>>>(cudaPrototype);
}

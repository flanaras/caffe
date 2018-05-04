/*#include <host_defines.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef nullptr
#define nullptr NULL
#endif

// TODO: check offloading communication control logic in gpu paper for synchronisation techniques.
// see cuStream{Write,Wait}Value32

enum status {none, waiting, requested, served, used};

struct data_exchange {
    //TODO: volatile variables
    status status;// = status::none;
    int requested_size;// = 0;
    int actual_size;// = 0;
    void* ptr;// = nullptr;
};

void empty_data_exchange(data_exchange *);

__device__ void blockUntil(data_exchange* dataExchange, status until) {
    while (true) {
        //TODO: read, with no race condition
        //TODO: CAS
        if (dataExchange->status == until) {
            return;
        }
    }
}

// From HOST to have a way to invoke data serving
__global__ void waitForDataRequest(data_exchange* dataExchange) {
    // TODO: CAS
    dataExchange->status = waiting;
    blockUntil(dataExchange, requested);
}

// From HOST when data are served to the GPU
__global__ void dataResponse(data_exchange* requestMount) {
    // blockUntil(requestMount, status::waiting);
    // TODO: copy things

    requestMount->status = served;
    blockUntil(requestMount, used);
    return;
}

__device__ void requestData(data_exchange* requestMount) {
    if (threadIdx.x == 0) {
        //TODO: write, with no race condition cas
        requestMount->status = requested;
    }
}

__global__ void data_layer_gpu(data_exchange* requestMount, int* ptr, cudaStream_t stream) {
    //TODO: How many things to transfer
    //TODO: Can assume count == blockDim.x
    atomicAdd(&requestMount->requested_size, 1);
    __syncthreads();


    //cuStreamCreate(&stream);
    //cuStreamWriteValue32(stream, (CUdeviceptr)ptr, 42, CU_STREAM_WAIT_VALUE_GEQ);
    requestData(requestMount);
    __syncthreads();

    //TODO: add image on the ith position in the blob
    //TODO: pass on the blob / return?
}

void thread2(data_exchange* cudaPrototype) {
    //execute listener
    //1 per block
    bool running = true;

    while(true) {
        // Wait for signal
        waitForDataRequest<<<1, 1>>>(cudaPrototype);

        // Acquire data
        void ** data;
        //data = getData();

        // Push data GPU
        // TODO: copy things to gpu

        // Send response
        dataResponse<<<1, 1>>>(cudaPrototype);
    }
}

int main() {
    // Init
    data_exchange* prototype = nullptr;
    data_exchange* cudaPrototype = nullptr;
    prototype = (data_exchange*) malloc(sizeof(data_exchange));
    CUDA_CHECK(cudaMalloc((void **)&cudaPrototype, sizeof(data_exchange)));
    empty_data_exchange(prototype);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemcpy(cudaPrototype, prototype, sizeof(prototype), cudaMemcpyDefault));

    //CAFFE_GET_BLOCKS(2);
    //CAFFE_CUDA_NUM_THREADS;

    int num_of_warps;

    for (int i = 0; i < num_of_warps; ++i) {
        // Todo: spawn threads
        thread2(cudaPrototype);
    }

    data_layer_gpu<<<1, 32>>>(cudaPrototype, (int*)cudaPrototype, stream);
}

void empty_data_exchange(data_exchange* data_exchage) {
    data_exchage->requested_size = 0;
    data_exchage->actual_size = 0;
    data_exchage->status = none;
    data_exchage->ptr = nullptr;
}*/

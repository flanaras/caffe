#include <stdint.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

using namespace std;

#define Dtype float

__global__ void foo(char *data, Dtype* mean, Dtype* transformed_data) {

  int baseptr = 227 * 227 * 3 * blockIdx.x;
 //154604 size
  Dtype datum_element = 0;
  int crop_size = 227;
  int top_index, data_index;
  int height = crop_size;
  int width = crop_size;
  int datum_channels = 3;
  int datum_height = crop_size;
  int datum_width = crop_size;
  Dtype scale = 1;
  bool do_mirror = !false;
  bool has_uint8 = true;
  bool has_mean_file = true;
  int h_off = 0;
  int w_off = 0;
  h_off = (datum_height - crop_size) / 2;
  w_off = (datum_width - crop_size) / 2;

  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w + baseptr;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w) + baseptr;
        } else {
          top_index = (c * height + h) * width + w + baseptr;
        }
        if (has_uint8) {
          datum_element = (Dtype)((uint8_t)data[data_index]);
        } else {
//          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
        } else {
//          if (has_mean_values) {
//            transformed_data[top_index] =
//                (datum_element - mean_values_[c]) * scale;
//          } else {
//            transformed_data[top_index] = datum_element * scale;
//          }
        }
      }
    }
  }
}

void runfoo() {
  int batch_size = 256 * 2 * 4;
  int data_size = 154604;
  char *data = new char[227 * 227 * 3 * batch_size];
  Dtype *mean = new Dtype[227 * 227 * 3 * batch_size];
  Dtype *tdata = new Dtype[227 * 227 * 3 * batch_size];
  char *gdata;
  Dtype *gmean;
  Dtype *gtdata;
  cudaError_t error;
  error = cudaMalloc((void**)&gdata, sizeof(char) * 227 * 227 * 3 * batch_size);
  cout << " " << cudaGetErrorString(error) << endl;

  error = cudaMalloc((void**)&gmean, sizeof(Dtype) * 227 * 227 * 3 * batch_size);
  cout << " " << cudaGetErrorString(error) << endl;

  error = cudaMalloc((void**)&gtdata, sizeof(Dtype) * 227 * 227 * 3 * batch_size);
  cout << " " << cudaGetErrorString(error) << endl;

  error = cudaMemcpy(gdata, data, 227 * 227 * 3 * batch_size * sizeof(char), cudaMemcpyDefault);
  cout << " " << cudaGetErrorString(error) << endl;

  error = cudaMemcpy(gmean, mean, 227 * 227 * 3 * batch_size * sizeof(Dtype), cudaMemcpyDefault);
  cout << " " << cudaGetErrorString(error) << endl;

  error = cudaMemcpy(gtdata, tdata, 227 * 227 * 3 * batch_size * sizeof(Dtype), cudaMemcpyDefault);
  cout << " " << cudaGetErrorString(error) << endl;

  for (int i = 0; i < 5; ++i) {
    foo <<< batch_size, 1  >>> (gdata, gmean, gtdata);
    error = cudaPeekAtLastError();
  }
  cout << "S " << cudaGetErrorString(error) << endl;
  error = cudaMemcpy(tdata, gtdata, 227 * 227 * 3 * batch_size * sizeof(Dtype), cudaMemcpyDefault);
}

int main() {
  runfoo();
  cudaProfilerStop();
}

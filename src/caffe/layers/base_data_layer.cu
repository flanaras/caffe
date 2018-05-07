#include <vector>
#include <caffe/util/benchmark.hpp>
#include <boost/thread/shared_lock_guard.hpp>
#include <boost/thread/mutex.hpp>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {
  template <typename Dtype>
  class Handler {
  private:

    class BookKeeping {
      vector<int> shape;
    };

  public:
    boost::mutex mutex;
    vector<caffe::Batch<Dtype>*> batches;
    caffe::Batch *current;
    int pointer_in_batch;
    int super_batch_size;
    int batch_size;
    vector<int> shape_label;
    BookKeeping data;
    BookKeeping labels;

    Handler() {
      this->super_batch_size = caffe::BasePrefetchingDataLayer::batch_size_factor;
    }


    void* get_gpu_batch_pointer_data() {
      // Give batch point
      int size_of;
      int size_of_labels;
      int batch_size;
    }

    void* get_gpu_batch_pointer_labels() {
    }

    void data_to_gpu() {
//      cudaMemcpy(&top[0], src_ptr, batch_size * size_of, cudaMemcpyDefault);
//      cudaMemcpy(&top[1], src_ptr, batch_size * size_of_labels, cudaMemcpyDefault);
    }

    // TODO: keep track of GPU memory

    void next() {

    }

    template<typename Dtype>
    void set_super_batch(caffe::Batch<void*> *pBatch) {
      boost::mutex::scoped_lock lock(mutex);

      data.push_back(pBatch);
    }
  };

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Handler handler;

//  // Send data
//  prefetch_current_ = prefetch_full_.pop("Waiting for data");
//  handler.set_super_batch(prefetch_current_);
//  handler.data_to_gpu();

  top[0]->ReshapeLike(handler.batch_data_shape());
  top[0]->set_gpu_data(handler.get_gpu_batch_pointer_data());

  if (this->output_labels_) {
    top[1]->ReshapeLike(handler.batch_label_shape());
    top[1]->set_gpu_data(handler.get_gpu_batch_pointer_labels());
  }

  handler.next();
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu2(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//  CPUTimer timer;
//  timer.Start();
  boost::posix_time::ptime start_cpu_;
  boost::posix_time::ptime start_cpu_2;
  boost::posix_time::ptime stop_cpu_2;
  start_cpu_ = boost::posix_time::microsec_clock::local_time();
  start_cpu_2 = boost::posix_time::microsec_clock::local_time();
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  stop_cpu_2 = boost::posix_time::microsec_clock::local_time();
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
  LOG_IF(INFO, Caffe::root_solver()) << "data_layer-us: " << (stop_cpu_2- start_cpu_2).total_microseconds()
                                     << "," << (boost::posix_time::microsec_clock::local_time() -
          start_cpu_).total_microseconds();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe

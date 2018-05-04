#include <vector>
#include <caffe/util/benchmark.hpp>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
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

#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

#include <vector>
#include <deque>
#include <boost/thread/shared_lock_guard.hpp>
#include <boost/thread/mutex.hpp>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

  using std::vector;
  using std::deque;

  template <typename Dtype>
  class Batch {
  public:
    Blob<Dtype> data_, label_;
  };


  template <typename Dtype>
  class Handler {

    class BookKeeping {
    public:
      BookKeeping(int batch_size, int size_of_item, vector<int> shape) : size_of_item(size_of_item),
                                                                         shape(shape),
                                                                         batch_size(batch_size) {
      }

      const vector<int>& get_shape() {
        return shape;
      }

    protected:
      int size_of_batch() {
        return size_of_item * batch_size;
      }

      const unsigned int batch_size;
      const unsigned int size_of_item;
      vector<int> shape;
    };

    class GPUKeeper : public BookKeeping {
    public:
      GPUKeeper(int number_of_items, int batch_size, int size_of_item, vector<int> shape) : number_of_items(number_of_items),
                                                                                              BookKeeping(batch_size, size_of_item, shape) {

        unsigned int sum = number_of_items * size_of_item;
        // 227 * 227 * 3 * 256 * 4 * 4
        LOG(WARNING) << "    bs: " << this->batch_size << " noi: " << number_of_items << " soi: " << size_of_item;
        LOG(WARNING) << "    >> sum: " << sum;
        CUDA_CHECK(cudaMalloc(&gpu_ptr, sum));
        LOG(WARNING) << "    >> ptr: " << gpu_ptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&async1, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&async2, cudaEventDisableTiming));
      }

      ~GPUKeeper() {
        CUDA_CHECK(cudaFree(gpu_ptr));
      }

      Dtype* get_gpu_ptr(int batch_id) {
        LOG(WARNING) << "    >> ptr: " << gpu_ptr + batch_id * this->size_of_batch() / sizeof(Dtype);
        return gpu_ptr + static_cast<unsigned long int>(batch_id * this->size_of_batch() / sizeof(Dtype));
      }

      void move(void* ptr, int index, int count) {
        CUDA_CHECK(cudaMemcpy(get_gpu_ptr(index), ptr, count * this->size_of_item, cudaMemcpyDefault));
      }
      void moveAsync(void* ptr, int index, int count) {
        CUDA_CHECK(cudaMemcpyAsync(get_gpu_ptr(index), ptr, count * this->size_of_item, cudaMemcpyDefault, stream));
        if (index == 0) {
          CUDA_CHECK(cudaEventRecord(async1, stream));
        } else {
          CUDA_CHECK(cudaEventRecord(async2, stream));
        }
      }

      void waitForAsync1() {
        CUDA_CHECK(cudaEventSynchronize(async1));
      }

      void waitForAsync2() {
        CUDA_CHECK(cudaEventSynchronize(async2));
      }

    private:
      cudaEvent_t async1;
      cudaEvent_t async2;
      const int number_of_items;
      cudaStream_t stream;
      Dtype* gpu_ptr;
    };

  private:
    boost::mutex mutex;
    deque<caffe::Batch<Dtype>*> batches_data;
    deque<caffe::Batch<Dtype>*> batches_empty;
    caffe::Batch<Dtype> *next_batch;

    int pointer_in_super_batch;
    const int batch_size;

    int get_super_batch_size() {
      return batch_size * super_batch_factor;
    }

    GPUKeeper data;
    GPUKeeper labels;

    bool gpu_has;
    const int factor_in_gpu;

  public:
    static const int super_batch_factor = 4;
    static const int _factor_in_gpu = 2;

    Handler(int batch_size,
            vector<int> data_shape,
            int data_size_of,
            vector<int> labels_shape,
            int labels_size_of) : batch_size(batch_size),
                                  factor_in_gpu(_factor_in_gpu),
                                  data(super_batch_factor * batch_size * _factor_in_gpu, batch_size, data_size_of, data_shape),
                                  labels(super_batch_factor * batch_size * _factor_in_gpu, batch_size, labels_size_of, labels_shape) {
      pointer_in_super_batch = 0;
      gpu_has = false;
    }

    const vector<int>& get_batch_data_shape() {
      return data.get_shape();
    }

    const vector<int>& get_batch_labels_shape() {
      return labels.get_shape();
    }

    Dtype* get_batch_data_gpu_pointer_data() {
      LOG(INFO) << "DATA pointer in super batch: " << pointer_in_super_batch << " batchsize: " << batch_size;
      if (pointer_in_super_batch == 0) {
        data.waitForAsync1();
      } else if (pointer_in_super_batch == super_batch_factor) {
        data.waitForAsync2();
      }
      return data.get_gpu_ptr(pointer_in_super_batch);
    }

    Dtype* get_batch_data_gpu_pointer_labels() {
      if (pointer_in_super_batch == 0) {
        data.waitForAsync1();
      } else if (pointer_in_super_batch == super_batch_factor) {
        data.waitForAsync2();
      }
      return labels.get_gpu_ptr(pointer_in_super_batch);
    }

    void next() {
      pointer_in_super_batch++;
      pointer_in_super_batch %= super_batch_factor * factor_in_gpu;

      if (pointer_in_super_batch % super_batch_factor == 0) {
        boost::mutex::scoped_lock lock(mutex);
        batches_data.push_back(next_batch);

        if (batches_data.size() == 0) {
          LOG(ERROR) << ">>>>>>>>>>> end of batches";
          exit(0);
        }
        next_batch = batches_data[0];
        batches_data.pop_front();
        move_data_async();
      }
    }

    void set_super_batch(caffe::Batch<Dtype> *pBatch) {
      {
        boost::mutex::scoped_lock lock(mutex);
        batches_data.push_back(pBatch);
      }
      {
        boost::mutex::scoped_lock lock(mutex);
        if (!gpu_has && batches_data.size() == 2) {

          next_batch = batches_data[0];
          batches_data.pop_front();
          move_data(0);
          batches_data.push_back(next_batch);

          next_batch = batches_data[0];
          batches_data.pop_front();
          move_data_async();

          gpu_has = true;
        }
      }
    }

  private:
    void move_data(int offset) {
      LOG(INFO) << " ------------ Moving data";
      data.move(next_batch->data_.mutable_cpu_data(), offset, get_super_batch_size());
      labels.move(next_batch->label_.mutable_cpu_data(), offset, get_super_batch_size());
    }

    void move_data_async() {
      LOG(INFO) << " ------------ Moving data Async";
      data.moveAsync(next_batch->data_.mutable_cpu_data(), get_unused_offset(), get_super_batch_size());
      labels.moveAsync(next_batch->label_.mutable_cpu_data(), get_unused_offset(), get_super_batch_size());
    }

    int get_unused_offset() const {
      int offset;
      if (pointer_in_super_batch < super_batch_factor) {
        offset = super_batch_factor;
      } else {
        offset = 0;
      }
      return offset;
    }

  };

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};


template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  shared_ptr<Handler<Dtype> > handler_;
  vector<shared_ptr<Batch<Dtype> > > prefetch_;
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_

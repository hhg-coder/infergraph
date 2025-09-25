#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <list>
#include "tensor.hpp"


namespace kuiper_infer {

/**
 * 张量池：用于静态推理时预分配和复用中间张量内存
 */
class TensorPool {
 public:
  using TensorKey = std::tuple<uint32_t, uint32_t, uint32_t>; // (channels, rows, cols)

  // 获取一个指定shape的张量，优先复用池中空闲张量
  std::shared_ptr<Tensor<float>> Get(uint32_t c, uint32_t h, uint32_t w) {
    TensorKey key{c, h, w};
    auto& free_list = pool_[key];
    if (!free_list.empty()) {
      auto tensor = free_list.front();
      free_list.pop_front();
      return tensor;
    }
    // 没有可复用的，分配新张量
    return std::make_shared<Tensor<float>>(c, h, w);
  }

  // 归还张量到池中
  void Release(const std::shared_ptr<Tensor<float>>& tensor) {
    if (!tensor) return;
    TensorKey key{tensor->channels(), tensor->rows(), tensor->cols()};
    pool_[key].push_back(tensor);
  }

  // 预分配指定shape和数量的张量
  void PreAllocate(uint32_t c, uint32_t h, uint32_t w, size_t count) {
    TensorKey key{c, h, w};
    auto& free_list = pool_[key];
    for (size_t i = 0; i < count; ++i) {
      free_list.push_back(std::make_shared<Tensor<float>>(c, h, w));
    }
  }

  // 清空池
  void Clear() {
    pool_.clear();
  }

 private:
  // key: (channels, rows, cols), value: 可复用张量列表
  std::unordered_map<TensorKey, std::list<std::shared_ptr<Tensor<float>>>, 
    std::hash<TensorKey>> pool_;
};

// TensorKey 的 hash 实现
} // namespace kuiper_infer

// hash specialization for TensorKey
namespace std {
template <>
struct hash<kuiper_infer::TensorPool::TensorKey> {
  size_t operator()(const kuiper_infer::TensorPool::TensorKey& key) const {
    auto h1 = std::hash<uint32_t>{}(std::get<0>(key));
    auto h2 = std::hash<uint32_t>{}(std::get<1>(key));
    auto h3 = std::hash<uint32_t>{}(std::get<2>(key));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};
}
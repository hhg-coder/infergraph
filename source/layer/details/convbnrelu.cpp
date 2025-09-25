#include "convbnrelu.hpp"
#include <glog/logging.h>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include <cmath>

namespace kuiper_infer {

ConvBNReLULayer::ConvBNReLULayer(uint32_t output_channel, uint32_t in_channel,
                                 uint32_t kernel_h, uint32_t kernel_w,
                                 uint32_t padding_h, uint32_t padding_w,
                                 uint32_t stride_h, uint32_t stride_w,
                                 uint32_t groups, bool use_bias)
    : ParamLayer("ConvBNReLU"),
      use_bias_(use_bias),
      groups_(groups),
      padding_h_(padding_h),
      padding_w_(padding_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  if (groups != 1) {
    in_channel /= groups;
  }
  this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
  if (use_bias_) {
    this->InitBiasParam(output_channel, 1, 1, 1);
  }
}

InferStatus ConvBNReLULayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  // 1. 卷积
  // 2. BN
  // 3. ReLU
  // 代码结构与ConvolutionLayer类似，省略部分参数检查...

  const uint32_t kernel_count = this->weights_.size();
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;
  const uint32_t kernel_count_group = kernel_count / groups_;
  const uint32_t batch_size = inputs.size();

  if (kernel_matrix_arr_.empty()) {
    this->InitIm2ColWeight();
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    const uint32_t input_c = input->channels();
    const uint32_t input_padded_h = input->rows() + 2 * padding_h_;
    const uint32_t input_padded_w = input->cols() + 2 * padding_w_;
    const uint32_t output_h =
        std::floor((int(input_padded_h) - int(kernel_h)) / stride_h_ + 1);
    const uint32_t output_w =
        std::floor((int(input_padded_w) - int(kernel_w)) / stride_w_ + 1);

    uint32_t col_len = output_h * output_w;
    uint32_t input_c_group = input_c / groups_;

    for (uint32_t g = 0; g < groups_; ++g) {
      const auto& input_matrix =
          Im2Col(input, kernel_w, kernel_h, input->cols(), input->rows(),
                 input_c_group, g, row_len, col_len);
      std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
      if (output_tensor == nullptr || output_tensor->empty()) {
        output_tensor =
            std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        outputs.at(i) = output_tensor;
      }

      const uint32_t kernel_count_group_start = kernel_count_group * g;
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        arma::frowvec kernel;
        if (groups_ == 1) {
          kernel = kernel_matrix_arr_.at(k);
        } else {
          kernel = kernel_matrix_arr_.at(kernel_count_group_start + k);
        }
        ConvGemmBias(input_matrix, output_tensor, g, k, kernel_count_group,
                     kernel, output_w, output_h);
      }
    }

    // BN + ReLU
    for (uint32_t c = 0; c < kernel_count; ++c) {
      float mean = bn_mean_.empty() ? 0.f : bn_mean_[c];
      float var = bn_var_.empty() ? 1.f : bn_var_[c];
      float gamma = bn_gamma_.empty() ? 1.f : bn_gamma_[c];
      float beta = bn_beta_.empty() ? 0.f : bn_beta_[c];
      float eps = bn_eps_;

      float* data_ptr = outputs.at(i)->matrix_raw_ptr(c);
      for (uint32_t idx = 0; idx < output_h * output_w; ++idx) {
        // BN
        float norm = (data_ptr[idx] - mean) / std::sqrt(var + eps);
        data_ptr[idx] = gamma * norm + beta;
        // ReLU
        if (data_ptr[idx] < 0) data_ptr[idx] = 0;
      }
    }
  }
  return InferStatus::kInferSuccess;
}

arma::fmat ConvBNReLULayer::Im2Col(sftensor input, uint32_t kernel_w,
                                   uint32_t kernel_h, uint32_t input_w,
                                   uint32_t input_h, uint32_t input_c_group,
                                   uint32_t group, uint32_t row_len,
                                   uint32_t col_len) const {
  arma::fmat input_matrix(input_c_group * row_len, col_len);
  const uint32_t input_padded_h = input_h + 2 * padding_h_;
  const uint32_t input_padded_w = input_w + 2 * padding_w_;
  const float padding_value = 0.f;
  for (uint32_t ic = 0; ic < input_c_group; ++ic) {
    float* input_channel_ptr =
        input->matrix_raw_ptr(ic + group * input_c_group);
    uint32_t current_col = 0;
    uint32_t channel_row = ic * row_len;
    for (uint32_t w = 0; w < input_padded_w - kernel_w + 1; w += stride_w_) {
      for (uint32_t r = 0; r < input_padded_h - kernel_h + 1; r += stride_h_) {
        float* input_matrix_ptr =
            input_matrix.colptr(current_col) + channel_row;
        current_col += 1;
        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
          const uint32_t region_w = input_h * (w + kw - padding_w_);
          for (uint32_t kh = 0; kh < kernel_h; ++kh) {
            if ((kh + r >= padding_h_ && kw + w >= padding_w_) &&
                (kh + r < input_h + padding_h_ &&
                 kw + w < input_w + padding_w_)) {
              float* region_ptr =
                  input_channel_ptr + region_w + (r + kh - padding_h_);
              *input_matrix_ptr = *region_ptr;
            } else {
              *input_matrix_ptr = padding_value;
            }
            input_matrix_ptr += 1;
          }
        }
      }
    }
  }
  return input_matrix;
}

void ConvBNReLULayer::ConvGemmBias(
    const arma::fmat& input_matrix, sftensor output_tensor, uint32_t group,
    uint32_t kernel_index, uint32_t kernel_count_group,
    const arma::frowvec& kernel, uint32_t output_w, uint32_t output_h) const {
  arma::fmat output(
      output_tensor->matrix_raw_ptr(kernel_index + group * kernel_count_group),
      output_h, output_w, false, true);

  if (!this->bias_.empty() && this->use_bias_) {
    std::shared_ptr<Tensor<float>> bias;
    bias = this->bias_.at(kernel_index);
    if (bias != nullptr && !bias->empty()) {
      float bias_value = bias->index(0);
      output = kernel * input_matrix + bias_value;
    }
  } else {
    output = kernel * input_matrix;
  }
}

void ConvBNReLULayer::InitIm2ColWeight() {
  const uint32_t kernel_count = this->weights_.size();
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;

  if (groups_ == 1) {
    const uint32_t kernel_count_group = kernel_count / groups_;
    std::vector<arma::frowvec> kernel_matrix_arr(kernel_count_group);
    arma::frowvec kernel_matrix_c(row_len * kernel_c);
    for (uint32_t k = 0; k < kernel_count_group; ++k) {
      const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
      for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
        memcpy(kernel_matrix_c.memptr() + row_len * ic,
               kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
      }
      kernel_matrix_arr.at(k) = kernel_matrix_c;
    }
    this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
  } else {
    const uint32_t kernel_count_group = kernel_count / groups_;
    std::vector<arma::frowvec> kernel_matrix_arr;
    for (uint32_t g = 0; g < groups_; ++g) {
      arma::fmat kernel_matrix_c(1, row_len * kernel_c);
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        const std::shared_ptr<Tensor<float>>& kernel =
            this->weights_.at(k + g * kernel_count_group);
        for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
          memcpy(kernel_matrix_c.memptr() + row_len * ic,
                 kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
        }
        kernel_matrix_arr.emplace_back(kernel_matrix_c);
      }
    }
    this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
  }
}

ParseParameterAttrStatus ConvBNReLULayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& cbr_layer) {
  // 参数解析流程与卷积类似，需额外解析BN参数
  CHECK(op != nullptr) << "ConvBNReLU operator is nullptr";
  const auto& params = op->params;

  // ...参数检查与卷积类似，略...

  // 这里只做简化演示，实际应完整检查
  auto in_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
  auto out_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
  auto kernel =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_size"));
  auto padding =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
  auto stride =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
  auto groups =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
  auto use_bias =
      std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));

  cbr_layer = std::make_shared<ConvBNReLULayer>(
      out_channel->value, in_channel->value, kernel->value.at(0), kernel->value.at(1),
      padding->value.at(0), padding->value.at(1), stride->value.at(0), stride->value.at(1),
      groups->value, use_bias->value);

  // 权重
  const auto& attrs = op->attribute;
  if (use_bias->value) {
    const auto& bias = attrs.at("bias");
    const std::vector<float>& bias_values = bias->get<float>();
    cbr_layer->set_bias(bias_values);
  }
  const auto& weight = attrs.at("weight");
  const std::vector<float>& weight_values = weight->get<float>();
  cbr_layer->set_weights(weight_values);

  // BN参数
  auto cbr_layer_derived = std::dynamic_pointer_cast<ConvBNReLULayer>(cbr_layer);
  cbr_layer_derived->bn_mean_ = attrs.at("bn_mean")->get<float>();
  cbr_layer_derived->bn_var_ = attrs.at("bn_var")->get<float>();
  cbr_layer_derived->bn_gamma_ = attrs.at("bn_gamma")->get<float>();
  cbr_layer_derived->bn_beta_ = attrs.at("bn_beta")->get<float>();
  // eps可根据模型参数调整
  cbr_layer_derived->bn_eps_ = 1e-5f;

  cbr_layer_derived->InitIm2ColWeight();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

// 注册
LayerRegistererWrapper kCBRGetInstance("nn.ConvBNReLU",
                                       ConvBNReLULayer::GetInstance);

}  // namespace kuiper_infer
#include "ml/tensor.h"
#include <iostream>

namespace ml {

Tensor::Tensor(std::vector<size_t> shape) : shape_(shape) {
    size_t total = 1;
    for (size_t dim : shape) total *= dim;
    data_.resize(total, 0.0f);
}

Tensor::Tensor(std::vector<size_t> shape, float fill_value) : shape_(shape) {
    size_t total = 1;
    for (size_t dim : shape) total *= dim;
    data_.resize(total, fill_value);
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = other.shape_[1];

    Tensor result({m, n});
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0;
            for (size_t l = 0; l < k; ++l) {
                sum += data_[i * k + l] * other.data_[l * n + j];
            }
            result.data_[i * n + j] = sum;
        }
    }
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose only supported for 2D tensors");
    }
    Tensor result({shape_[1], shape_[0]});
    for (size_t i = 0; i < shape_[0]; ++i) {
        for (size_t j = 0; j < shape_[1]; ++j) {
            result.data_[j * shape_[0] + i] = data_[i * shape_[1] + j];
        }
    }
    return result;
}

float Tensor::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0f);
}

float Tensor::mean() const {
    return sum() / data_.size();
}

void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}

}  // namespace ml

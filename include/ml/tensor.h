#pragma once

#include <vector>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <iostream>

namespace ml {

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, float fill_value);

    size_t size() const { return data_.size(); }
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }

    float& operator[](size_t index) { return data_[index]; }
    const float& operator[](size_t index) const { return data_[index]; }

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;

    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor reshape(std::vector<size_t> new_shape) const;

    float sum() const;
    float mean() const;
    float max() const;
    float min() const;

    void fill(float value);
    void print() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
};

}  // namespace ml

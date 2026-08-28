#pragma once

#include <cmath>
#include <algorithm>
#include <numeric>

namespace ml {

enum class LossType {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy
};

class Loss {
public:
    virtual ~Loss() = default;
    virtual float compute(const Tensor& predicted, const Tensor& actual) = 0;
    virtual Tensor gradient(const Tensor& predicted, const Tensor& actual) = 0;
};

class MSELoss : public Loss {
public:
    float compute(const Tensor& predicted, const Tensor& actual) override;
    Tensor gradient(const Tensor& predicted, const Tensor& actual) override;
};

class CrossEntropyLoss : public Loss {
public:
    float compute(const Tensor& predicted, const Tensor& actual) override;
    Tensor gradient(const Tensor& predicted, const Tensor& actual) override;
};

}  // namespace ml

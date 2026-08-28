#pragma once

#include "tensor.h"
#include <functional>
#include <string>

namespace ml {

enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradients, float learning_rate) = 0;
    virtual std::string name() const = 0;
};

class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradients, float learning_rate) override;
    std::string name() const override { return "Dense"; }

private:
    Tensor weights_;
    Tensor biases_;
    Tensor last_input_;
};

class ActivationLayer : public Layer {
public:
    ActivationLayer(ActivationType type);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& gradients, float learning_rate) override;
    std::string name() const override { return "Activation"; }

private:
    ActivationType type_;
    Tensor last_input_;
};

}  // namespace ml

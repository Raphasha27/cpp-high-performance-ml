#pragma once

#include "tensor.h"
#include <vector>
#include <memory>

namespace ml {

class NeuralNetwork {
public:
    NeuralNetwork() = default;
    ~NeuralNetwork() = default;

    void addLayer(std::unique_ptr<Layer> layer);
    void compile(float learning_rate = 0.01);
    
    Tensor forward(const Tensor& input);
    void backward(const Tensor& gradients);
    
    void train(const Tensor& X, const Tensor& y, int epochs, int batch_size = 32);
    float evaluate(const Tensor& X, const Tensor& y);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    float learning_rate_ = 0.01;
};

}  // namespace ml

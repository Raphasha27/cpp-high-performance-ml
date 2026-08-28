#include "ml/tensor.h"
#include "ml/neural_network.h"
#include "ml/layer.h"
#include "ml/loss.h"
#include <iostream>

int main() {
    std::cout << "XOR Problem - Neural Network from Scratch" << std::endl;
    std::cout << "==========================================" << std::endl;

    ml::NeuralNetwork nn;
    nn.addLayer(std::make_unique<ml::DenseLayer>(2, 8));
    nn.addLayer(std::make_unique<ml::ActivationLayer>(ml::ActivationType::ReLU));
    nn.addLayer(std::make_unique<ml::DenseLayer>(8, 1));
    nn.addLayer(std::make_unique<ml::ActivationLayer>(ml::ActivationType::Sigmoid));
    nn.compile(0.1);

    ml::Tensor X({4, 2});
    X[0] = 0; X[1] = 0;
    X[2] = 0; X[3] = 1;
    X[4] = 1; X[5] = 0;
    X[6] = 1; X[7] = 1;

    ml::Tensor y({4, 1});
    y[0] = 0;
    y[1] = 1;
    y[2] = 1;
    y[3] = 0;

    std::cout << "\nTraining..." << std::endl;
    nn.train(X, y, 1000, 4);

    ml::Tensor output = nn.forward(X);
    std::cout << "\nPredictions:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: [" << X[i*2] << ", " << X[i*2+1] 
                  << "] -> Output: " << output[i] << std::endl;
    }

    return 0;
}

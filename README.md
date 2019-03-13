# neural-networks

This project provides neural networks learning based on back propagation algorithm. 

Basically, a neural network system consists of nodes and weights. Nodes and weights are created as classes. Weight classes store how related between nodes. Some network attributes are defined in main program such as hidden layer size and how many nodes exist in hidden layer. 

The most optimum weight values should be picked up to have best neural network model. In order to pick up optimum weights; forward propagation, backpropagation and stockastic gradient descent algorithms are applied respectively. Firsyly, forward propagation algorithm multiplies input and weights, and network output is calculated. Secondly, back propagation algorithm is applied to calculate node errors based on assigned weigts (initially weights are randomly assigned). Thirdly, stockastic gradient descent is applied to update weights. Finally, these process (forward prop, back prop, grad desc) are applied again and again. Thus, optimum weights could be picked up and error is minimized. Cost variable is dumped in main program to monitor error change in each gradient descent iteration

The math behind the back propagation algorithm is explained in the following link: https://serengil.wordpress.com/2017/01/21/the-math-behind-backpropagation/

Usage
=====

Run the Backpropagation.java under the package com.ml.nn. The default program uses the historical data under the dataset folder. 

You could use your own dataset and run backpropagation algorithm. Basically, each line stands for historical data instance. In a line, attributes are seperated by commas. Importantly, the final element in a line is actual output, and other elements are input variables. e.g. in xor.txt example, input 1 and input 2 are 0 whereas output is 0 in first line.

Also, you could change some variables in main program such as hidden nodes, learning rate, epoch.

License
=======

Chefboost is licensed under the MIT License - see [LICENSE](https://github.com/serengil/neural-networks/blob/master/LICENSE) for more details.

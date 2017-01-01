# neural-networks

This project provides neural networks learning based on back propagation algorithm. 

Basically, a neural network system consists of nodes and weights. Nodes and weights are created as classes. Weight classes store how related between nodes. Some network attributes are defined in main program such as hidden layer size and how many nodes exist in hidden layer. 

The most optimum weight values should be picked up to have best neural network model. In order to pick up optimum weights; forward propagation, backpropagation and stockastic gradient descent algorithms are applied respectively. Firsyly, forward propagation algorithm multiplies input and weights, and network output is calculated. Secondly, back propagation algorithm is applied to calculate node errors based on assigned weigts (initially weights are randomly assigned). Thirdly, stockastic gradient descent is applied to update weights. Finally, these process (forward prop, back prop, grad desc) are applied again and again. Thus, optimum weights could be picked up and error is minimized. Cost variable is dumped in main program to monitor error change in each gradient descent iteration

=====
Usage
=====

Run the Backpropagation.java under the package com.ml.nn. The default program uses the historical data under the dataset folder. 

You could use your own dataset and run backpropagation algorithm. Basically, each line stands for historical data instance. In a line, attributes are seperated by commas. Importantly, the final element in a line is actual output, and other elements are input variables. e.g. in xor.txt example, input 1 and input 2 are 0 whereas output is 0 in first line.

Also, you could change some variables in main program such as hidden nodes, learning rate, epoch.

=======
License
=======

Copyright 2016 Sefik Ilkin Serengil

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

# neural-networks

This project provides neural networks learning based on back propagation algorithm. 

Basically, a neural network system consists of nodes and weights. Nodes and weights are created as classes. Weight classes store how related between nodes. Some network attributes as defined in main program such as hidden layer size and how many nodes exist in hidden layer. Initially, weight values are randomly set. 

Back propagation algorithm is applied to calculate node errors based on assigned weigts. This requires to apply forward propation first. After than, stockastic gradient descent is applied to update weights. Also, cost is calculated for each back propagation iteration.

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

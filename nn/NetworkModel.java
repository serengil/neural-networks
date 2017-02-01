package com.ml.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.ml.nn.entity.Node;
import com.ml.nn.entity.Weight;

public class NetworkModel {
	
public static List<Node> createNodes(int numberOfInputs, int[] hiddenNodes, boolean dump){
		
		if(dump)
			System.out.println("network model...");
		
		List<Node> nodes = new ArrayList<Node>();
		
		int nodeIndex = 0;
		
		//------------------------------------
		//input layer
		
		if(dump)
			System.out.print("Input layer:  ");
		
		//bias unit
		Node biasNode = new Node();
		biasNode.setLevel(0);
		biasNode.setItem(0);
		biasNode.setLabel("+1");
		biasNode.setLayerName("Input layer");
		biasNode.setIndex(nodeIndex);
		biasNode.setBiasUnit(true);
		nodes.add(biasNode);
		
		if(dump)
			System.out.print(biasNode.getLabel()+"\t");
		
		nodeIndex++; //increase for bias unit
		
		//bias unit end
		
		for(int i=0;i<numberOfInputs;i++){
			
			if(dump)
				System.out.print("V"+(i+1)+"\t"); //variable
			
			Node node = new Node();
			node.setLevel(0);
			node.setItem(i+1);
			node.setLabel("V"+(i+1)); //variable
			node.setLayerName("Input layer");
			node.setIndex(nodeIndex);
			node.setBiasUnit(false);
			nodes.add(node);
			
			nodeIndex++;
			
		}
		
		if(dump)
			System.out.println();
		
		//------------------------------------
		//hidden layer
		
		for(int i=0;i<hiddenNodes.length;i++){
			
			if(dump)
				System.out.print("Hidden layer: ");
			
			//bias unit
			biasNode = new Node();
			biasNode.setLevel(i+1);
			biasNode.setItem(0);
			biasNode.setLabel("+1");
			biasNode.setLayerName("Hidden layer ("+(i+1)+")");
			biasNode.setIndex(nodeIndex);
			biasNode.setBiasUnit(true);
			nodes.add(biasNode);
			
			if(dump)
				System.out.print(biasNode.getLabel()+"\t");
			
			nodeIndex++;
			
			//bias unit end
			
			for(int j=0;j<hiddenNodes[i];j++){
				
				Node node = new Node();
				node.setLevel(i+1);
				node.setItem(j+1);
				node.setLabel("N["+(i+1)+", "+(j+1)+"]");
				node.setLayerName("Hidden layer ("+(i+1)+")");
				node.setIndex(nodeIndex);
				node.setBiasUnit(false);
				nodes.add(node);
				
				if(dump)
					System.out.print("N["+(i+1)+", "+(j+1)+"]\t\t");
				
				nodeIndex++;
				
			}
			
			if(dump)
				System.out.println();
			
		}
		
		//output layer
		
		Node node = new Node();
		node.setLevel(1 + hiddenNodes.length);
		node.setItem(1);
		node.setLabel("Output");
		node.setLayerName("Output layer ");
		node.setIndex(nodeIndex);
		nodes.add(node);
		
		if(dump){
			System.out.println("Output layer: Output\n");
		}
		
		return nodes;
		
	}

	public static List<Weight> createWeights(List<Node> nodes, int numberOfInputs, int[] hiddenNodes, boolean dump){
	
	if(dump)
		System.out.println("connection creation with random weights...");
	
	List<Weight> weights = new ArrayList<Weight>();
	
	int totalLayers = 1 + hiddenNodes.length + 1; //input layer + hidden layers + output layer
	
	double randomValue = 0;
	
	int weightIndex = 0;
	
	for(int i=0;i<totalLayers - 1;i++){
		
		for(int j=0;j<nodes.size();j++){
			
			if(nodes.get(j).getLevel() == i){
				
				//i level item: nodes.get(j).getLabel()
				
				for(int k=0;k<nodes.size();k++){
					
					if(nodes.get(k).getLevel() == i+1){
						
						if(nodes.get(k).isBiasUnit() == false){
							
							//there is a connection from nodes.get(j).getLabel() to nodes.get(k).getLabel()
							
							//randomly initialize weights, then these weights will be updated by backpropagation algoritm
							//initialize all weights between [-epsilon, +epsilon]. 
							//the following paper gives opinion about initializing. https://web.stanford.edu/class/ee373b/nninitialization.pdf
							
							Random r = new Random();
							double rangeMin = 0, rangeMax = 1;
							double INIT_EPSILON = (double) Math.sqrt(6) / Math.sqrt(numberOfInputs + 1);
							double rand = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
							randomValue = rand * (2 * INIT_EPSILON) - INIT_EPSILON;
														
							//------------------------------------------
							
							Weight weight = new Weight();
							weight.setWeightIndex(weightIndex);
							weight.setFromIndex(nodes.get(j).getIndex());
							weight.setFromLabel(nodes.get(j).getLabel());
							weight.setToIndex(nodes.get(k).getIndex());
							weight.setToLabel(nodes.get(k).getLabel());		
							weight.setValue(randomValue); //initialize each weight between [-epsilon, +epsilon]
							weights.add(weight);
							
							weightIndex++;
							
							if(dump)
								System.out.println("from "+nodes.get(j).getLabel()+" to "+nodes.get(k).getLabel()+": "+randomValue);
							
						}	
						
					}
					
				}
				
			}
			
		}
		
	}
	
	return weights;
}

}

package com.ml.nn;

/**
 * @author Sefik Ilkin Serengil
 * 
 * initialization: 2017-01-01
 * lastly updated: 2017-01-30
 * 
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.math.BigDecimal;
import com.ml.nn.entity.Attribute;
import com.ml.nn.entity.BackProp;
import com.ml.nn.entity.HistoricalItem;
import com.ml.nn.entity.Node;
import com.ml.nn.entity.Weight;

public class Backpropagation {
	
	public static boolean dump = true;
	
	public static void main(String[] args) {
				
		//variable definition
		
		int[] hiddenNodes = {4}; //a hidden layer consisting of 4 nodes. 
		//to create a networkconsisting of multiple hidden layers define the variable as {3, 3}. 
		//this usage means 2 hidden layers consisting of 3 nodes for each layer
		
		String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\sine.txt";
		//String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\xor.txt";
		
		double bias = 1;
		double learningRate = 0.01; //learning rate should be between 0 and 1, mostly less than or equal to 0.2 (Alpaydin, E., 2004)
		double momentum = 0;
		int epoch = 100000; //the larger epoch, the better learning
		
		String activation = "sigmoid"; //available functions: sigmoid, tanh, softsign, gaussian
		
		System.out.println("activation function: "+activation);
		
		//------------------------------------------------
		
		//load trainingset		
		List<HistoricalItem> historicalData = HistoricalData.retrieveHistoricalData(historicalDataPath);
		int numberOfInputs = historicalData.get(0).getAttributes().size() - 1; //final item is output, the others are attributes
		
		//normalize inputs
		List<HistoricalItem> attributeBoundaries = findAttributeBoundaries(historicalData);
		historicalData = normalizeAttributes(historicalData, attributeBoundaries, activation);
		
		//node creation
		List<Node> nodes = createNodes(numberOfInputs, hiddenNodes, dump);
		
		//weight creation
		List<Weight> weights = createWeights(nodes, numberOfInputs, hiddenNodes, dump); 
		
		//store cost after each gradient descent iteration
		List<Double> costs = new ArrayList<Double>();
		
		if(dump)
			System.out.println("\nCosts after gradient descent iterations...");
		
		for(int i=0;i<=epoch;i++){			

			//apply back propagation
			BackProp backProp = applyBackPropagation(historicalData, nodes, weights, activation, learningRate, momentum, bias, false);		
			nodes = backProp.getNodes();
			weights = backProp.getWeights();
			
			//calculate cost
			double J = calculateCost(historicalData, nodes, weights, activation, bias, false);
			costs.add(J);
			
			//display costs for each iteration
			if(dump && i % 10000 == 0){
				System.out.println(i+"\t"+new BigDecimal(J));
			}
						
		}
		
		//display final weights
		System.out.println("\nfinal weights...");
		for(int i=0;i<weights.size();i++){
			
			System.out.println("from "+weights.get(i).getFromLabel()+" to "+weights.get(i).getToLabel()+": "+weights.get(i).getValue()); 
			
		}
		
		//display predictions on dataset
		System.out.println("\nfinal outputs...\nactual\tpredict");
		for(int i=0;i<historicalData.size();i++){
			
			List<Node> currentNodes = applyForwardPropagation(historicalData.get(i), nodes, weights, activation, bias, false);
				
			double min = attributeBoundaries.get(attributeBoundaries.size()-1).getAttributes().get(0).getValue();
			double max = attributeBoundaries.get(attributeBoundaries.size()-1).getAttributes().get(1).getValue();
			
			double normalizedMin = Activation.retrieveRange(activation, "min", "output");
			double normalizedMax = Activation.retrieveRange(activation, "max", "output");
			
			if(dump){
				System.out.println(
							denormalizeAttribute(historicalData.get(i).getAttributes().get(historicalData.get(i).getAttributes().size()-1).getValue(), max, min, normalizedMax, normalizedMin) //actual
							+"\t"
							+denormalizeAttribute(currentNodes.get(currentNodes.size()-1).getValue(), max, min, normalizedMax, normalizedMin) //predict
						);	
			}
			
		}		
				
	}
	
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
	
	public static List<Node> applyForwardPropagation(HistoricalItem instance, List<Node> nodes, List<Weight> weights, String activation, double bias, boolean dump){
		
		//transfer bias unit values first		
		for(int j=0;j<nodes.size();j++){
			
			if(nodes.get(j).isBiasUnit() == true){
				
				nodes.get(j).setValue(bias);
				
			}
			
		}
		
		//transfer historical item to input layer (sigmoid function will not be applied for inputs)
		for(int j=0;j<instance.getAttributes().size()-1;j++){ //final item is output, that is why size - 1
			
			//String key = instance.getAttributes().get(j).getKey();
			double var = instance.getAttributes().get(j).getValue();
			
			for(int k=0;k<nodes.size();k++){
				
				if(j+1 == nodes.get(k).getIndex()){
					
					nodes.get(k).setValue(var);
					break;
					
				}
				
			}

		}
		//transfer historical item end
		
		for(int j=0;j<nodes.size();j++){
			
			if(nodes.get(j).getLevel() > 0 && nodes.get(j).isBiasUnit() == false){ //input layer discarded
				
				double netinput = 0, netoutput = 0;
				
				int targetIndex = nodes.get(j).getIndex();
				
				for(int k=0;k<weights.size();k++){
					
					if(targetIndex == weights.get(k).getToIndex()){
						
						double wi = weights.get(k).getValue();
						
						double sourceIndex = weights.get(k).getFromIndex();
						
						for(int m=0;m<nodes.size();m++){
							
							if(sourceIndex == nodes.get(m).getIndex()){
								
								double xi = nodes.get(m).getValue();
								
								netinput = netinput + (double) (xi*wi);
								
								break;
								
							}
							
						}
						
					}
					
				}
				
				netoutput = Activation.activationFunction(activation, netinput);				
				nodes.get(j).setNetInputValue(netinput); //store net input value
				nodes.get(j).setValue(netoutput); //store net output value

			} //input layer discarded checking end
			
		}
		
		if(dump){			
			System.out.println(instance.getAttributes().get(instance.getAttributes().size()-1).getValue()+"\t"+nodes.get(nodes.size()-1).getValue());	
		}
		
		return nodes;
		
	}
	
	public static BackProp applyBackPropagation(List<HistoricalItem> historicalData, List<Node> nodes, List<Weight> weights, String activation, double learningRate, double momentum, double bias, boolean dump){
		
		//this block includes forward propagation, back propagation and stockastic gradient descent
		
		if(dump)
			System.out.println("applying back propagation...");
		
		int numberOfInputsAttributes = historicalData.get(0).getAttributes().size() - 1; //final item is output, the others are attributes
		
		for(int i=0;i<historicalData.size();i++){
			
			//apply forward propagation first
			nodes = applyForwardPropagation(historicalData.get(i), nodes, weights, activation, bias, dump);
			
			//historical instance
		
			Node outputNode = nodes.get(nodes.size()-1);
			
			double actualValue = historicalData.get(i).getAttributes().get(numberOfInputsAttributes).getValue();
			double predictValue = outputNode.getValue();
			
			double smallDelta = actualValue - predictValue;
			
			nodes.get(nodes.size()-1).setSmallDelta(smallDelta);
			
			for(int j=nodes.size()-2;j>numberOfInputsAttributes;j--){ //output delta already calculated on the step above. that is why nodes.size - 2
				
				//look for connections including from nodes.get(j)
				
				int targetIndex = nodes.get(j).getIndex();
				
				double sumOfSmallDelta = 0;
				
				for(int k=0;k<weights.size();k++){
					
					if(weights.get(k).getFromIndex() == targetIndex){
						
						double affectingTheta = weights.get(k).getValue();
						double affectingSmallDelta = 1;
						
						int targetSmallDeltaIndex = weights.get(k).getToIndex();
						
						for(int m=0;m<nodes.size();m++){
							
							if(nodes.get(m).getIndex() == targetSmallDeltaIndex){
								
								affectingSmallDelta = nodes.get(m).getSmallDelta();
								
								break;
								
							}
							
						}
						
						double newlySmallDelta = affectingTheta * affectingSmallDelta;
						
						sumOfSmallDelta = sumOfSmallDelta + newlySmallDelta;
						
					}
					
				}
				
				nodes.get(j).setSmallDelta(sumOfSmallDelta);
								
			} //calculation of small deltas end
			
			//---------------------------------

			//apply stockastic gradient descent to update weights
			
			double previousDerivative = 0;
			
			for(int j=0;j<weights.size();j++){
				
				double weightFromNodeValue = 0, weightToNodeDelta = 0, weightToNodeValue = 0, weightToNodeNetInput = 0;
				
				for(int k=0;k<nodes.size();k++){
					
					if(nodes.get(k).getIndex() == weights.get(j).getFromIndex()){
						
						weightFromNodeValue = nodes.get(k).getValue();
						
					}
					
					if(nodes.get(k).getIndex() == weights.get(j).getToIndex()){
						
						weightToNodeDelta = nodes.get(k).getSmallDelta();
						weightToNodeValue = nodes.get(k).getValue();
						weightToNodeNetInput = nodes.get(k).getNetInputValue();
						
					}
					
				}
				
				double derivative = weightToNodeDelta 
						* Activation.derivativeOfActivation(activation, weightToNodeValue, weightToNodeNetInput) 
						* weightFromNodeValue;
				
				//weights.get(j).setValue(weights.get(j).getValue() + learningRate * derivative); //without momentum
				weights.get(j).setValue(weights.get(j).getValue() + learningRate * ( derivative + momentum * previousDerivative) ); //momentum capability added
				previousDerivative = derivative * 1;
				
			}
			
			//weight update end
			
		} //historical instance loop end
		
		BackProp backProp = new BackProp();
		backProp.setNodes(nodes);
		backProp.setWeights(weights);
		
		return backProp;
		
	}
	
	public static List<HistoricalItem> findAttributeBoundaries(List<HistoricalItem> historicalData){
		
		List<HistoricalItem> datasetMinMax = new ArrayList<HistoricalItem>();
		
		for(int k=0;k<historicalData.get(0).getAttributes().size();k++){
			
			HistoricalItem attributeMinMax = new HistoricalItem();
			List<Attribute> minMaxInstance = new ArrayList<Attribute>();
			
			double maxItem = -10000, minItem = 10000;
			
			//find max and min elements
			for(int i=0;i<historicalData.size();i++){
				
				double output = historicalData.get(i).getAttributes().get(historicalData.get(i).getAttributes().size()-1).getValue();
				
				if(output < minItem)
					minItem = output;
				
				if(output > maxItem)
					maxItem = output;
				
			}
			
			Attribute minValue = new Attribute();
			minValue.setKey("min");
			minValue.setValue(minItem);
			minMaxInstance.add(minValue);
			
			Attribute maxValue = new Attribute();
			maxValue.setKey("max");
			maxValue.setValue(maxItem);
			minMaxInstance.add(maxValue);
			
			attributeMinMax.setAttributes(minMaxInstance);
			
			datasetMinMax.add(attributeMinMax);
			
		}
		
		return datasetMinMax;
		
	}
	
	public static List<HistoricalItem> normalizeAttributes(List<HistoricalItem> historicalData, List<HistoricalItem> datasetMinMax, String activation){

		for(int i=0;i<historicalData.size();i++){
			
			for(int j=0;j<historicalData.get(i).getAttributes().size();j++){
				
				double newMin = 0, newMax = 0;
				
				if(j == historicalData.get(i).getAttributes().size() - 1){ //output item
					
					newMin = Activation.retrieveRange(activation, "min", "output");
					newMax = Activation.retrieveRange(activation, "max", "output");
					
				}
				else{ //input variable
					
					newMin = Activation.retrieveRange(activation, "min", "input");
					newMin = Activation.retrieveRange(activation, "max", "input");
					
				}
				
				double value = historicalData.get(i).getAttributes().get(j).getValue();
				double minItem = datasetMinMax.get(j).getAttributes().get(0).getValue();
				double maxItem = datasetMinMax.get(j).getAttributes().get(1).getValue();
				
				double normalizeValue = ((newMax - newMin)*((value - minItem) / (maxItem - minItem))) + newMin; //this line normalizes in scale [newMin, newMax]
				
				historicalData.get(i).getAttributes().get(j).setValue(normalizeValue);
				
			}
			
		}
		
		return historicalData;
		
	}
	
	public static double denormalizeAttribute(double normalizedValue, double max, double min, double normalizedMax, double normalizedMin){
		
		return (( (normalizedValue - normalizedMin) / (normalizedMax - normalizedMin) ) * (max - min)) + min; //denormalized values in scale [normalizedMin, normalizedMax]
		
	}
	
	public static  double calculateCost(List<HistoricalItem> historicalData, List<Node> nodes, List<Weight> weights, String activation, double bias, boolean dump){
		
		double J = 0;
		
		for(int i=0;i<historicalData.size();i++){
			
			nodes = applyForwardPropagation(historicalData.get(i), nodes, weights, activation, bias, dump);
			
			double predict = nodes.get(nodes.size() - 1).getValue();
			double actual = historicalData.get(i).getAttributes().get(historicalData.get(i).getAttributes().size()-1).getValue();
			
			double cost = (predict - actual)*(predict - actual);
			cost = cost / 2;
			
			J = J + cost;
			
		}
		
		J = J / historicalData.size();
		
		return J;
		
	}
	
}

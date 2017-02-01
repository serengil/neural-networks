package com.ml.nn;

/**
 * @author Sefik Ilkin Serengil
 * 
 * initialization: 2017-01-01
 * lastly updated: 2017-02-01
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
	public static double bias = 1;
	
	public static void main(String[] args) {
				
		//variable definition
		
		int[] hiddenNodes = {4}; //a hidden layer consisting of 4 nodes. 
		//to create a networkconsisting of multiple hidden layers define the variable as {3, 3}. 
		//this usage means 2 hidden layers consisting of 3 nodes for each layer
		
		String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\sine.txt";
		//String historicalDataPath = System.getProperty("user.dir")+"\\dataset\\xor.txt";
		
		double learningRate = 0.01; //learning rate should be between 0 and 1, mostly less than or equal to 0.2 (Alpaydin, E., 2004)
		double momentum = 0;
		int epoch = 3000; //the larger epoch, the better learning
		
		String activation = "sigmoid"; //available functions: sigmoid, tanh, softsign, gaussian
		
		System.out.println("activation function: "+activation);
		
		//------------------------------------------------
		
		//load trainingset		
		List<HistoricalItem> historicalData = HistoricalData.retrieveHistoricalData(historicalDataPath);
		int numberOfInputs = historicalData.get(0).getAttributes().size() - 1; //final item is output, the others are attributes
		
		//normalize inputs
		List<HistoricalItem> attributeBoundaries = Normalizing.findAttributeBoundaries(historicalData);
		historicalData = Normalizing.normalizeAttributes(historicalData, attributeBoundaries, activation);
		
		//node creation
		List<Node> nodes = NetworkModel.createNodes(numberOfInputs, hiddenNodes, dump);
		
		//weight creation
		List<Weight> weights = NetworkModel.createWeights(nodes, numberOfInputs, hiddenNodes, dump); 
		
		//store cost after each gradient descent iteration
		List<Double> costs = new ArrayList<Double>();
		
		if(dump)
			System.out.println("\nCosts after gradient descent iterations...");
		
		for(int i=0;i<=epoch;i++){			

			//apply back propagation
			BackProp backProp = NetworkLearning.applyBackPropagation(historicalData, nodes, weights, activation, learningRate, momentum, bias, false);		
			nodes = backProp.getNodes();
			weights = backProp.getWeights();
			
			//calculate cost
			double J = NetworkLearning.calculateCost(historicalData, nodes, weights, activation, bias, false);
			costs.add(J);
			
			//display costs for each iteration
			if(dump && i % 25 == 0){
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
			
			List<Node> currentNodes = NetworkLearning.applyForwardPropagation(historicalData.get(i), nodes, weights, activation, bias, false);
				
			double min = attributeBoundaries.get(attributeBoundaries.size()-1).getAttributes().get(0).getValue();
			double max = attributeBoundaries.get(attributeBoundaries.size()-1).getAttributes().get(1).getValue();
			
			double normalizedMin = Activation.retrieveRange(activation, "min", "output");
			double normalizedMax = Activation.retrieveRange(activation, "max", "output");
			
			if(dump){
				System.out.println(
							Normalizing.denormalizeAttribute(historicalData.get(i).getAttributes().get(historicalData.get(i).getAttributes().size()-1).getValue(), max, min, normalizedMax, normalizedMin) //actual
							+"\t"
							+Normalizing.denormalizeAttribute(currentNodes.get(currentNodes.size()-1).getValue(), max, min, normalizedMax, normalizedMin) //predict
						);	
			}
			
		}		
				
	}
	
}

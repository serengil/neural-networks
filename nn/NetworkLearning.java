package com.ml.nn;

import java.util.List;

import com.ml.nn.entity.BackProp;
import com.ml.nn.entity.HistoricalItem;
import com.ml.nn.entity.Node;
import com.ml.nn.entity.Weight;

public class NetworkLearning {
	
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
				weights.get(j).setValue(
						weights.get(j).getValue() 
						+ learningRate * derivative 
						+ momentum * previousDerivative 
					); //momentum capability added
				previousDerivative = derivative * 1;
				
			}
			
			//weight update end
			
		} //historical instance loop end
		
		BackProp backProp = new BackProp();
		backProp.setNodes(nodes);
		backProp.setWeights(weights);
		
		return backProp;
		
	}
	
	public static  double calculateCost(List<HistoricalItem> historicalData, List<Node> nodes, List<Weight> weights, String activation, double bias, boolean dump){
		
		double J = 0;
		
		for(int i=0;i<historicalData.size();i++){
			
			nodes = NetworkLearning.applyForwardPropagation(historicalData.get(i), nodes, weights, activation, bias, dump);
			
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

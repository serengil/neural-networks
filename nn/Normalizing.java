package com.ml.nn;

import java.util.ArrayList;
import java.util.List;

import com.ml.nn.entity.Attribute;
import com.ml.nn.entity.HistoricalItem;

public class Normalizing {
	
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

}

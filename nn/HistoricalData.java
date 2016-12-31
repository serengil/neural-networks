package com.ml.nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import com.ml.nn.entity.Attribute;
import com.ml.nn.entity.HistoricalItem;

public class HistoricalData {
	
	public static List<HistoricalItem> retrieveHistoricalData(String filepath){
		
		List<HistoricalItem> historicalData = new ArrayList<HistoricalItem>();

		try{
						
			BufferedReader br = new BufferedReader(new FileReader(filepath));
			
			String line = br.readLine();
			
			String header = "";
			header = ""+line;
			
			while (line != null) {
				
		        line = br.readLine();
		        
		        if(line != null){
		        
			        String headerlabels[] = header.split(",");
			        String[] items = line.split(",");
			        
			        List<Attribute> attributes = new ArrayList<Attribute>();
			        	
			        HistoricalItem historicalItem = new HistoricalItem();
			        	
			        for(int i=0;i<items.length;i++){
			        		
			        	Attribute attribute = new Attribute(); 
			        		
			        	String key = headerlabels[i];
			        	double value = Double.parseDouble(items[i]);
			        		
			        	attribute.setKey(key);
			        	attribute.setValue(value);
			        		
			        	attributes.add(attribute);
			        		
			        	historicalItem.setAttributes(attributes);
			        		
			        }
			        	
			        historicalData.add(historicalItem);
			        
		        }
   
		    }
			
			br.close();
			
			System.out.println("Historical data consisting of "+historicalData.size()+" items retrieved\n");

		}
		catch(Exception ex){
			
			System.out.println(ex);
			
		}
		
		return historicalData;
		
	}

}

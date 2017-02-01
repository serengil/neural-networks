package com.ml.nn;

public class Activation {
	
	public static double retrieveRange(String function, String direction, String attribute){
		
		double normalizedMin = 0, normalizedMax = 0;
		
		if("output".equals(attribute)){
			
			if("sigmoid".equals(function)){
				normalizedMin = 0;	
				normalizedMax = 1;
			}
			else if("tanh".equals(function)){
				normalizedMin = -1;
				normalizedMax = 1;
			}
			else if("softsign".equals(function)){
				normalizedMin = -1;
				normalizedMax = 1;
			}
			else if("gaussian".equals(function)){
				normalizedMin = 0;
				normalizedMax = 1;
			}			
			
		}
		else if("input".equals(attribute)){
			
			if("sigmoid".equals(function)){ //normalize in scale [-4,+4]
				normalizedMin = -4;
				normalizedMax = 4;
			}
			else if("tanh".equals(function)){ //normalize in scale [-2,+2]
				normalizedMin = -2;
				normalizedMax = 2;
			}
			else if("softsign".equals(function)){
				normalizedMin = -4;
				normalizedMax = 4;
			}
			else if("gaussian".equals(function)){
				normalizedMin = -1;
				normalizedMax = 1;
			}
			
		}
		
		if("min".equals(direction)){
			return normalizedMin;
		}
		else if("max".equals(direction)){
			return normalizedMax;
		}
		
		return 0;
		
	}
	
	public static double activationFunction(String function, double x){
		
		double f = 0;
		
		if("sigmoid".equals(function)){
			f = 1 / (1 + (Math.exp(-x)));
		}
		else if("tanh".equals(function)){
			f = (Math.exp(x) - Math.exp(-x))/(Math.exp(x) + Math.exp(-x));
		}
		else if("softsign".equals(function)){
			f = x / (1 + Math.abs(x));
		}
		else if("gaussian".equals(function)){
			f = Math.pow(Math.E, (-1)*x*x);
		}
		
		return f;
		
	}
	
	public static double derivativeOfActivation(String function, double fx, double x){
		
		double d = 0;
		
		if("sigmoid".equals(function)){
			d = fx * (1 - fx);
		}
		else if("tanh".equals(function)){
			d = 1 - (fx * fx);			
		}
		else if("softsign".equals(function)){
			d = 1 / ((1+Math.abs(x))*(1+Math.abs(x)));
		}
		else if("gaussian".equals(function)){
			d = -2*x*Math.pow(Math.E, (-1)*x*x);
		}
		
		return d;
		
	}

}

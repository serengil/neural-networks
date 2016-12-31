package com.ml.nn.entity;

import java.util.List;

public class CostEntity {
	
	List<Weight> weights;
	public double cost;
	
	public List<Weight> getWeights() {
		return weights;
	}
	public void setWeights(List<Weight> weights) {
		this.weights = weights;
	}
	public double getCost() {
		return cost;
	}
	public void setCost(double cost) {
		this.cost = cost;
	}
	
}

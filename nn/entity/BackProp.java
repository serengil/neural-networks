package com.ml.nn.entity;

import java.util.List;

public class BackProp {
	
	List<Node> nodes;
	List<Weight> weights;
	
	public List<Node> getNodes() {
		return nodes;
	}
	public void setNodes(List<Node> nodes) {
		this.nodes = nodes;
	}
	public List<Weight> getWeights() {
		return weights;
	}
	public void setWeights(List<Weight> weights) {
		this.weights = weights;
	}
	
}

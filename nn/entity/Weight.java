package com.ml.nn.entity;

public class Weight {
	
	public int weightIndex;
	public int fromIndex;
	public int toIndex;
	public String fromLabel;
	public String toLabel;
	public double value;
	public double capitalDelta;
	public double derivative;
	
	public int getFromIndex() {
		return fromIndex;
	}
	public void setFromIndex(int fromIndex) {
		this.fromIndex = fromIndex;
	}
	public int getToIndex() {
		return toIndex;
	}
	public void setToIndex(int toIndex) {
		this.toIndex = toIndex;
	}
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}
	public String getFromLabel() {
		return fromLabel;
	}
	public void setFromLabel(String fromLabel) {
		this.fromLabel = fromLabel;
	}
	public String getToLabel() {
		return toLabel;
	}
	public void setToLabel(String toLabel) {
		this.toLabel = toLabel;
	}
	public double getDerivative() {
		return derivative;
	}
	public void setDerivative(double derivative) {
		this.derivative = derivative;
	}
	public double getCapitalDelta() {
		return capitalDelta;
	}
	public void setCapitalDelta(double capitalDelta) {
		this.capitalDelta = capitalDelta;
	}
	public int getWeightIndex() {
		return weightIndex;
	}
	public void setWeightIndex(int weightIndex) {
		this.weightIndex = weightIndex;
	}

}

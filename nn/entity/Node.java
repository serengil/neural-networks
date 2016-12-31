package com.ml.nn.entity;

public class Node {
	
	public int index;
	public int level;
	public int item;
	public String label;
	public String layerName;
	public double value;
	public double smallDelta;
	public boolean biasUnit;
	
	public int getLevel() {
		return level;
	}
	public void setLevel(int level) {
		this.level = level;
	}
	public int getItem() {
		return item;
	}
	public void setItem(int item) {
		this.item = item;
	}
	public String getLabel() {
		return label;
	}
	public void setLabel(String label) {
		this.label = label;
	}
	public String getLayerName() {
		return layerName;
	}
	public void setLayerName(String layerName) {
		this.layerName = layerName;
	}
	public int getIndex() {
		return index;
	}
	public void setIndex(int index) {
		this.index = index;
	}
	public double getValue() {
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}
	public double getSmallDelta() {
		return smallDelta;
	}
	public void setSmallDelta(double smallDelta) {
		this.smallDelta = smallDelta;
	}
	public boolean isBiasUnit() {
		return biasUnit;
	}
	public void setBiasUnit(boolean biasUnit) {
		this.biasUnit = biasUnit;
	}

	
	
}

package org.makingstan.components;

import org.makingstan.Network;

public class Weight {

    //layer neuron previousNeuron
    public double[][][] weights;

    Network network;
    
    public Weight(Network network)
    {
        this.network = network;
        weights = new double[network.getNetworkSize()][][];
    }

    public double getWeight(int layer, int neuron, int previousNeuron)
    {
        return weights[layer][neuron][previousNeuron];
    }

    public void setWeight(int layer, int neuron, int previousNeuron, double value)
    {
        weights[layer][neuron][previousNeuron] = value;
    }

    public void addValueToAWeight(int layer, int neuron, int previousNeuron, double value)
    {
        weights[layer][neuron][previousNeuron] += value;
    }

    public void importWeight(double[][][] importedWeight)
    {
        weights = importedWeight;
    }
}

package org.makingstan.components;

import org.makingstan.Network;

public class Bias {

    //layer neuron
    public double[][] biases;

    public Bias(Network network)
    {
        this.biases = new double[network.getNetworkSize()][];
    }

    public void addBias(int layer, int neuron, double value)
    {
        biases[layer][neuron] = value;
    }

    public double getBias(int layer, int neuron)
    {
        return biases[layer][neuron];
    }

    public void addValueToABias(int layer, int neuron, double value)
    {
        biases[layer][neuron] += value;
    }

    public void importBias(double[][] bias)
    {
        biases = bias;
    }
}

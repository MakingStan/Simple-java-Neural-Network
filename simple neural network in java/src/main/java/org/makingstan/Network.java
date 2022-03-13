/*
initial code by finn eggers
https://www.youtube.com/watch?v=d3OtgsGcMLw&list=PLgomWLYGNl1dL1Qsmgumhcg4HOcWZMd3k
optimized and made more clear and explained in code by Stan Mijten

start date: 7 march 17:03 2022
 */
package org.makingstan;

import org.makingstan.components.*;

import java.io.*;
import java.util.Arrays;
import java.util.Random;


//implementing Serializable is needed to be able to save and load are network. see more below or in the main function
public class Network implements Serializable{

    /*
    BEFORE NOTICE
    i would only recommend looking at this code if you know how a basic neural network works
    not exactly the whole math behind it (it would be a bit handy tho)
    but just the concept on how it works
    otherwise you will have a really hard time understanding this code

    im also expecting some knowledge of java.
    if you don't know how multidimensional arrays work and some basic concepts
    this code will be hard to understand

    ----------MENTION----------
    i am a complete beginner in machine learning and if anyone knows something
    i explain better in this code please make an issue.
    I just explain this because it helps me even furthur expand my knowledge of machine learning
    And to help other complete beginners who know how a
    neural network works but have no idea how to put it in practice in code
     */

    //layer neuron
    private final Bias biases;
    //layer neuron previousNeuron
    private final Weight weights;

    private final int[] networkLayerSizes;
    private final int networkSize;
    private final int inputSize;
    private final int outputSize;

    private final double[][] output;
    //the diffrence between
    private final double[][] outputDerative;
    //layer neuron.
    private final double[][] costFunctionOutput;

    Random random = new Random();




    public Network(int... networkLayerSizes)
    {
        this.networkLayerSizes = networkLayerSizes;
        this.networkSize = networkLayerSizes.length;
        this.inputSize = networkLayerSizes[0];
        this.outputSize = networkLayerSizes[networkSize-1];

        this.output = new double[networkSize][];
        this.outputDerative = new double[networkSize][];
        this.costFunctionOutput = new double[networkSize][];

        biases = new Bias(this);
        weights = new Weight(this);

        //give it some random bias and weight  to start with.

        for(int i = 0; i < networkSize; i++)
        {
            //we have to do this because it fixes an error. otherwise everything in our arrays will be null pretty straight foreward stuff
            this.costFunctionOutput[i] = new double[networkLayerSizes[i]];
            this.output[i] = new double[networkLayerSizes[i]];
            this.outputDerative[i] = new double[networkLayerSizes[i]];

            biases.biases[i] = Tools.createRandomArray(networkLayerSizes[i], -0.5, 0.7);

            //arrays can't be in the minus and a weight needs the previousNeuron
            if(i > 0)
            {
                weights.weights[i] = Tools.createRandomArray(networkLayerSizes[i],networkLayerSizes[i-1], -0.3, 0.5);
            }

        }
    }



    //train function (void function because we don't have to return anything)
    public void train(double[] input, double[] target, double learningRate)
    {
        if(input.length != inputSize || target.length != outputSize) return;

        //calculate function we will use the data it sets in variables later
        calculate(input);
        //cost function
        cost(target);
        //update the weights
        updateWeightsAndBiases(learningRate);

    }

    // it's a void function because we don't have to return anything (updateWeights function)
    //learningRate could also sometimes be called eta (i think atleast :))
    private void updateWeightsAndBiases(double learningRate)
    {
        //loop through every layer we don't need to update the bias on the first layer because they don't have one.
        // only from middle and beyond they will have one. if we need the value
        // of the first layer we will just do layer -1 (index 0) is the first layer
        for(int layer = 1; layer < networkSize; layer++)
        {
            //loop through every neuron
            for (int neuron = 0; neuron < networkLayerSizes[layer]; neuron++)
            {
                //loop through every previousNeuron.
                for (int previousNeuron = 0; previousNeuron < networkLayerSizes[layer - 1]; previousNeuron++)
                {
                    //some math with gradient descent (i clearly don't know alot about this math.
                    // if you want to understand it i got an article here https://en.wikipedia.org/wiki/Delta_rule

                    /*
                    short explained
                    we get multiply the negative learning rate by the output of the previousNeuron and we do that times
                    our costfunction output which if you want to see how that is calculated go to the cost function
                     */
                    double delta = - learningRate * output[layer -1][previousNeuron] * costFunctionOutput[layer][neuron];

                    //just add the value to the weight( can be minus or plus)
                    //if you don't get this you'll probaly know what i mean with the line below
                    //weights[layer, neuron, previousNeuron] += delta
                    weights.addValueToAWeight(layer, neuron, previousNeuron, delta);
                }
                //do the same stuff with the biases
                double delta = -learningRate * costFunctionOutput[layer][neuron];
                biases.addValueToABias(layer, neuron, delta);
            }
        }
    }

    //again void function because we don't have to return anything (cost function)
    public void cost(double[] target)
    {
        /*this is the backpropagation we have to do this to be able to let the network actually "learn"
        i also at the most trouble understanding this part.
        so please if anyone knows something is wrongly explained. make a github issue.
        */

        //iterate through every neuron.
        for(int neuron = 0; neuron < networkLayerSizes[networkSize-1]; neuron++)
        {
            //set the costFunction output on the latest layer
            costFunctionOutput[networkSize-1][neuron] =
                    (output[networkSize-1][neuron] - target[neuron])
                            * outputDerative[networkSize-1][neuron];
        }

        //iterate through every layer from the last layer to the second (i think)
        for(int layer = networkSize-2; layer > 0; layer--)
        {
            //iterate through every neuron
            for(int neuron = 0; neuron < networkLayerSizes[layer]; neuron++)
            {
                double sum = 0;
                //iterate through every neuron destination of the weight
                for(int nextNeuron = 0; nextNeuron < networkLayerSizes[layer+1]; nextNeuron++)
                {
                    /*we get the weight on the last layer (we do +1 because we start with -2)
                    we get the weight that is connected to the nextneuron and the neuron we are currently on
                    and we multiply this by te costfunctionoutput on on that weight
                     */
                    sum += weights.getWeight(layer+1, nextNeuron, neuron) * costFunctionOutput[layer+1][nextNeuron];
                }

                //set the costFunctionOutput to our sum times our outputderative.
                this.costFunctionOutput[layer][neuron] = sum* outputDerative[layer][neuron];
            }
        }
    }







    /*makes any number a number between 0 and 1 (elu is more efficient but ill just use this)
    the y is the output. lookup some signmoid graphs and you'll see what it means
    i think there is a function called elu which does the same but a little bit different. i have heard it is better for machine learning but for now we will just stick with sigmoid
    picture of this here: https://en.wikipedia.org/wiki/Sigmoid_function
    (for example 0 will be 0.5, -2 will be something like 0.1)
    */
    private double sigmoid(double x)
    {
        return 1d/ (1 + Math.exp(-x));
    }

    public double[] calculate(double... input)
    {
        //to much or to fewer data to calculate anything
        if (input.length != this.inputSize) return null;

        output[0] = input;

        //iterate through the layers we start at one because we can easily access the
        // layer before us with doing layer -1 and the first row does not need any calculations
        for(int layer = 1; layer < networkSize; layer++)
        {
            //only iterate through how much neurons you have on the network in that specific layer
            for(int neuron = 0; neuron < networkLayerSizes[layer]; neuron++)
            {
                double sum = biases.getBias(layer, neuron);

                //loop our previous neurons
                for(int previousNeuron = 0; previousNeuron < networkLayerSizes[layer -1]; previousNeuron++)
                {
                    //calculate a neuron
                    //first we pick the previousNeuron from the last layer
                    //we multiply that by the weights of our connection
                    sum += output[layer - 1][previousNeuron] * weights.getWeight(layer, neuron, previousNeuron);
                }
                //now we sigmoid our sum. this just makes the number a number between 0 and 1
                output[layer][neuron] = sigmoid(sum);
                //calculate the difference so we can adjust the weights and the biases after. (we only use this while training)
                outputDerative[layer][neuron] = sigmoid(output[layer][neuron]);

            }
        }
        //output the last layer of the network. this is the output. the network size. is one longer then that so that's why we use -1
        return output[networkSize-1];
    }



    public static void main(String[] args) throws Exception {
        /*
        lets make an example function that when put in 1 the output will be 1 or close to
        one and when it's 0 the output will be 0 or close to zero. extremely simple but
        remember that we didn't tell the computer what a 1 or a 0 is
        this is just purely from our network

        you can do much complexer things with this network this is just a simple example
         */
        Network network = new Network(1, 2, 1);

        //epochs is basically how much times you want to train


        int epochs = 1000;

        for(int i = 0; i < epochs; i++)
        {
            /*generate a random number and set this to the input and output
            it will generate a number 1 or 0. we set this to the input and our target.
            this basically means that we want 1 as an output if it's 1 as an input same the other way around.
             */
            int randomNumber = network.random.nextInt(2);

            double[] input = new double[]{randomNumber};
            double[] target = new double[]{randomNumber};
            /*learning rate 0.3 seems to be a good balance. we don't want
            learningrate to be to high because otherwise it will be innacurate
            but also not to low because otherwise it will learn really slow
            */

            network.train(input, target, 1);
        }
         /*
            this way of loading and saving the network is not extremely good.
            but it works. the reason why it's not that good is if we change one thing about the code like make a new function
            this loading the network where that function wasn't there doesn't work.
            You can make a function that just saves the biases and the weights but that's up to you
         */


        //we save the network like this
        network.saveNetwork("src/main/resources/output.txt");



        //we load it like this. it's in a comment because we currently don't want to load the network
        network.loadNetwork("src/main/resources/output.txt");

        //bam it works! :)
        System.out.println(Arrays.toString(network.calculate(1)));

    }

    //get the input size
    public int getInputSize()
    {
        return inputSize;
    }

    //get the output size
    public int getOutputSize()
    {
        return outputSize;
    }

    //get the network size
    public int getNetworkSize()
    {
        return networkSize;
    }

    //get the networkLayerSizes variable
    public int[] getNetworkLayerSizes()
    {
        return networkLayerSizes;
    }


    //this way of saving the network really isn't good. because if we will change any and then any code to the network it won't load
    public void saveNetwork(String file) throws IOException
    {
        File f = new File(file);
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(f));

        out.writeObject(this);
        out.flush();
        out.close();
    }

    //doesn't work if first saved and then modified. if it's not modified loading should just work
    public Network loadNetwork(String file) throws Exception
    {
        File f = new File(file);
        ObjectInput input = new ObjectInputStream(new FileInputStream(f));

        Network network = (Network) input.readObject();

        return network;
    }


}

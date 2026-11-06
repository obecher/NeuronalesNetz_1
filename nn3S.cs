#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks_1
{
    class nn3S
    {
        double[,] wih, who;
        int inodes, hnodes, onodes;
        double[] hidden_inputs;
        double[] hidden_outputs;
        double[] final_inputs;
        double[] final_outputs;
        double[] hidden_errors;
double[] output_errors;
double learningRate;
nnMath nnMathO = new nnMath();

        public double[,] WIH { get { return wih; } }
        public double[,] WHO { get { return who; } }
        public double[] Hidden_inputs { get { return hidden_inputs; } }
        public double[] Hidden_outputs { get { return hidden_outputs; } }
        public double[] Final_inputs { get { return final_inputs; } }
        public double[] Final_outputs { get { return final_outputs; } }
        public double[] Hidden_errors { get { return hidden_errors; } }
public double[] Output_errors { get { return output_errors; } }
        
public nn3S(int inodes, int hnodes, int onodes, double learningRate)
{
    this.inodes = inodes;
    this.hnodes = hnodes;
    this.onodes = onodes;
    this.learningRate = learningRate;

    createWeightMatrizes();
}

        private void createWeightMatrizes()
        {
            wih = new double[inodes, hnodes];
            who = new double[hnodes, onodes];

            wih[0, 0] = 0.9; //spalte, reihe
            wih[1, 0] = 0.3;
            wih[2, 0] = 0.4;
            wih[0, 1] = 0.2;
            wih[1, 1] = 0.8;
            wih[2, 1] = 0.2;
            wih[0, 2] = 0.1;
            wih[1, 2] = 0.5;
            wih[2, 2] = 0.6;

            who[0, 0] = 0.3;
            who[1, 0] = 0.7;
            who[2, 0] = 0.5;
            who[0, 1] = 0.6;
            who[1, 1] = 0.5;
            who[2, 1] = 0.2;
            who[0, 2] = 0.8;
            who[1, 2] = 0.1;
            who[2, 2] = 0.9;
        }

        public void queryNN(double[] inputs)
        {
            nnMath nnMathO = new nnMath();

            hidden_inputs = new double[hnodes];
            hidden_inputs = nnMathO.matrixMult(wih, inodes, inputs);
           
            hidden_outputs = new double[hnodes];
            hidden_outputs = nnMathO.activationFunction(hidden_inputs);
           

            final_inputs = new double[onodes];
           final_inputs = nnMathO.matrixMult(who, hnodes, hidden_outputs);

            final_inputs = nnMathO.matrixMult(who, hnodes, hidden_outputs);
            final_outputs = nnMathO.activationFunction(final_inputs);
           
        }
        public void train(double[] inputs, double[] targets)
        {
            queryNN(inputs);

            output_errors = nnMathO.vectorSub(targets, final_outputs);
            double [,] who_transposed = nnMathO.transpose(who);
            hidden_errors = nnMathO.matrixMult(who_transposed, onodes, output_errors);
            double[] gradient_output = new double[onodes];
    for (int i = 0; i < onodes; i++)
    {
        gradient_output[i] = output_errors[i] * final_outputs[i] * (1.0 - final_outputs[i]);
    }

    // Äußeres Produkt bilden (multipliziert mit den Ausgaben der vorherigen Schicht)
    double[,] delta_who = nnMathO.fullMatrixMult(gradient_output, hidden_outputs);
    // Mit der Lernrate skalieren
    double[,] delta_who_scaled = nnMathO.matrixScale(delta_who, learningRate);
    // Zur bestehenden Matrix addieren
    who = nnMathO.matrixSum(who, delta_who_scaled);

    // 5. Gewichte zwischen Eingabeschicht und versteckter Schicht anpassen (WIH)
    // Formel: += learningRate * [(hidden_errors * hidden_outputs * (1.0 - hidden_outputs)) * (inputs)T]
    
    // Gradient für die versteckte Schicht berechnen
    double[] gradient_hidden = new double[hnodes];
    for (int i = 0; i < hnodes; i++)
    {
        gradient_hidden[i] = hidden_errors[i] * hidden_outputs[i] * (1.0 - hidden_outputs[i]);
    }

    // Äußeres Produkt bilden
    double[,] delta_wih = nnMathO.fullMatrixMult(gradient_hidden, inputs);
    // Mit der Lernrate skalieren
    double[,] delta_wih_scaled = nnMathO.matrixScale(delta_wih, learningRate);
    // Zur bestehenden Matrix addieren
    wih = nnMathO.matrixSum(wih, delta_wih_scaled);
        }
    }
    
}


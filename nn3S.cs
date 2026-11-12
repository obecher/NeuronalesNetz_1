#nullable disable
using System;

namespace NeuralNetworks_1
{
    // 3-Schichten-Netz (Input - Hidden - Output)
    // Konvention der Gewichtsmatrizen: [von-Knoten, nach-Knoten]
    //   wih[inodes, hnodes]   who[hnodes, onodes]
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
        public double LearningRate { get { return learningRate; } set { learningRate = value; } }

        public nn3S(int inodes, int hnodes, int onodes, double learningRate)
        {
            this.inodes = inodes;
            this.hnodes = hnodes;
            this.onodes = onodes;
            this.learningRate = learningRate;

            createWeightMatrizes();
        }

        // Startgewichte: Zufallszahlen gleichverteilt zwischen -1.0 und +1.0
        private void createWeightMatrizes()
        {
            wih = new double[inodes, hnodes];
            who = new double[hnodes, onodes];
            Random rnd = new Random();

            for (int i = 0; i < inodes; i++)
                for (int j = 0; j < hnodes; j++)
                    wih[i, j] = rnd.NextDouble() * 2.0 - 1.0;   // [0,1) -> [-1,1)

            for (int i = 0; i < hnodes; i++)
                for (int j = 0; j < onodes; j++)
                    who[i, j] = rnd.NextDouble() * 2.0 - 1.0;
        }

        // Zum Laden trainierter Gewichte von der Festplatte
        public void setWIHMatrix(double[,] matrix)
        {
            if (matrix.GetLength(0) != inodes || matrix.GetLength(1) != hnodes)
                throw new ArgumentException("WIH hat die falsche Größe");
            wih = matrix;
        }

        public void setWHOMatrix(double[,] matrix)
        {
            if (matrix.GetLength(0) != hnodes || matrix.GetLength(1) != onodes)
                throw new ArgumentException("WHO hat die falsche Größe");
            who = matrix;
        }

        public void queryNN(double[] inputs)
        {
            hidden_inputs = nnMathO.matrixMult(wih, inodes, inputs);
            hidden_outputs = nnMathO.activationFunction(hidden_inputs);

            final_inputs = nnMathO.matrixMult(who, hnodes, hidden_outputs);
            final_outputs = nnMathO.activationFunction(final_inputs);
        }

        public void train(double[] inputs, double[] targets)
        {
            // 1. Vorwärts rechnen
            queryNN(inputs);

            // 2. Fehler am Ausgang: e = target - output
            output_errors = nnMathO.vectorSub(targets, final_outputs);

            // 3. Fehler zurück in die versteckte Schicht verteilen: e_hidden = WHO * e_out
            //    (who ist [hnodes, onodes]; transponiert [onodes, hnodes] -> Ergebnis Länge hnodes)
            double[,] who_transposed = nnMathO.transpose(who);
            hidden_errors = nnMathO.matrixMult(who_transposed, onodes, output_errors);

            // 4. Gewichte anpassen (erst WHO, dann WIH)
            //    WICHTIG: hidden_errors wurden oben mit dem ALTEN who berechnet - so ist es richtig.
            updateWeights(who, output_errors, final_outputs, hidden_outputs);
            updateWeights(wih, hidden_errors, hidden_outputs, inputs);
        }

        // Gradientenabstieg für eine Gewichtsmatrix w[von, nach]:
        //   w[i,j] += learningRate * (e_j * o_j * (1 - o_j)) * prev_i
        // e = Fehler der "nach"-Schicht, o = Ausgaben der "nach"-Schicht, prev = Ausgaben der "von"-Schicht
        // Entspricht: w += lr * fullMatrixMult(prev, e*o*(1-o)), aber direkt in der Matrix
        // (ohne 3 neue 784x100-Matrizen pro Bild -> beim 60K-Training deutlich schneller)
        private void updateWeights(double[,] w, double[] errors, double[] outputs, double[] prevOutputs)
        {
            int rows = w.GetLength(0);
            int cols = w.GetLength(1);

            double[] gradient = new double[cols];
            for (int j = 0; j < cols; j++)
                gradient[j] = learningRate * errors[j] * outputs[j] * (1.0 - outputs[j]);

            for (int i = 0; i < rows; i++)
            {
                double p = prevOutputs[i];
                for (int j = 0; j < cols; j++)
                    w[i, j] += gradient[j] * p;
            }
        }
    }
}

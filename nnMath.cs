using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks_1;

class nnMath
{
    public double[] matrixMult(double[,] mat1, int mat1width, double[] mat2)
    {

        double[] result = new double[mat2.Length];
        for (int j = 0; j < result.Length; j++ )
        for (int i = 0; i < mat2.Length; i++)
        result[j] += mat1[i,j] * mat2[i];

        return result;
    }
    public double[] activationFunction(double[] inputs)
    {
        double[] result = new double[inputs.Length];

        for( int i = 0; i < inputs.Length; i++)
        result[i] = 1.0 / (1.0 + Math.Exp(-inputs[i]));
        return result;
    }
}
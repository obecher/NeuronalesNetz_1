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

    public double[,] fullMatrixMult(double[] vec1, double[] vec2)
    {
        double [,] result = new double[vec1.Length, vec2.Length];
        for (int i = 0; i < vec1.Length; i++)
        for (int j = 0; j < vec2.Length; j++)
        result[i, j] = vec1[i] * vec2[j];
        return result;
    }

    public double[,] matrixSum(double[,] mat1, double[,] mat2)
    {
        int rows = mat1.GetLength(0);
        int cols = mat1.GetLength(1);
    double[,] result = new double[rows, cols];
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i, j] = mat1[i, j] + mat2[i, j];
        }
    }
    
    return result;
}
public double[] vectorSub(double[] vec1, double[] vec2) 
{
    double[] result = new double[vec1.Length];
    for (int i = 0; i < vec1.Length; i++)
    {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}


public double[] vectorMult(double[] vec1, double[] vec2) 
{
    double[] result = new double[vec1.Length];
    for (int i = 0; i < vec1.Length; i++)
    {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}
// matrix trasnponieren
public double[,] transpose(double[,] inputMat) 
{
    int rows = inputMat.GetLength(0);
    int cols = inputMat.GetLength(1);
    
   
    double[,] result = new double[cols, rows];
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[j, i] = inputMat[i, j]; // i und j tauschen
        }
    }
    
    return result;
}
public double[,] matrixScale(double[,] inputMat, double scale) 
{
    int rows = inputMat.GetLength(0);
    int cols = inputMat.GetLength(1);
    double[,] result = new double[rows, cols];
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i, j] = inputMat[i, j] * scale;
        }
    }
    
    return result;
}

}

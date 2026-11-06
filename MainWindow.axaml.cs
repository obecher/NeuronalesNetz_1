using System;
using System.Collections.ObjectModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;

namespace NeuralNetworks_1;

public partial class MainWindow : Window
{
    int inodes = 3, hnodes = 3, onodes = 3;
    nn3S? nn3SO;
    double[] inputs = Array.Empty<double>();
    double learningRate = 0.1;

    // Avalonia's DataGrid has no Items.Add(), rows are added to the ItemsSource collection
    readonly ObservableCollection<nodeRow> rows = new();

    public MainWindow()
    {
        InitializeComponent();

        networkDataGrid.ItemsSource = rows;

        // WPF PreviewTextInput = Avalonia TextInput event in the Tunnel phase
        inputTextBox.AddHandler(TextInputEvent, inputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        hiddenTextBox.AddHandler(TextInputEvent, hiddenTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        outputTextBox.AddHandler(TextInputEvent, outputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
    }

    private void inputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

    private void hiddenTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

    private void outputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !int.TryParse(e.Text, out _);
    }

   private void createButton_Click(object? sender, RoutedEventArgs e)
    {
        if ((inodes != 0) && (hnodes != 0) && (onodes != 0))
            nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);
    }

    private void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null) return;

        inputs = new double[] { 0.9, 0.1, 0.8 };
        nn3SO.queryNN(inputs);
        
        displayResults(false); // false = nur Abfrage, kein Training
    }

    private void trainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null) return;

        inputs = new double[] { 0.9, 0.1, 0.8 };
        // Zielwerte für das Training festlegen (wie in der Anleitung gefordert)
        double[] targets = new double[] { 0.9, 0.9, 0.9 }; 

        nn3SO.train(inputs, targets);
        
        displayResults(true); // true = wir haben trainiert, Fehler anzeigen
    }

    // Eigene Methode zum Anzeigen, damit wir den Code nicht doppelt schreiben müssen
    private void displayResults(bool trained)
    {
        rows.Clear(); // Ganz wichtig: Alte Tabelle leeren!

        for (int i = 0; i < inputs.Length; i++)
        {
            var data = new nodeRow {
                inputValue = inputs[i].ToString(),
                weightsIH = nn3SO.WIH[0, i].ToString() + "," + nn3SO.WIH[1, i].ToString() + "," + nn3SO.WIH[2, i].ToString(),
                inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]),
                outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]),
                weightsHO = nn3SO.WHO[0, i].ToString() + "," + nn3SO.WHO[1, i].ToString() + "," + nn3SO.WHO[2, i].ToString(),
                
                // Wenn trainiert wurde, zeige den echten Fehler, sonst "0"
                errorHidden = trained ? String.Format(" {0:0.##} ", nn3SO.Hidden_errors[i]) : "0",
                
                inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]),
                outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]),
                
                // Wenn trainiert wurde, zeige Ziel und Ausgabe-Fehler
                target = trained ? "0.9" : "0",
                errorOutput = trained ? String.Format(" {0:0.##} ", nn3SO.Output_errors[i]) : "0"
            };
            rows.Add(data);}
    }
} 
        public class nodeRow
{
    public string inputValue { get; set; } = "";
    public string weightsIH { get; set; } = "";
    public string inputHidden { get; set; } = "";
    public string outputHidden { get; set; } = "";
    public string weightsHO { get; set; } = "";
    public string errorHidden { get; set; } = "";
    public string inputOutput { get; set; } = "";
    public string outputLayer { get; set; } = "";
    public string target { get; set; } = "";
    public string errorOutput { get; set; } = "";
}
    
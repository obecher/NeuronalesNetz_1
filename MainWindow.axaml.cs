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
            nn3SO = new nn3S(inodes, hnodes, onodes);
    }

    private void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null)
            return;

        int i;
        inputs = new double[inodes];

        inputs[0] = 0.9;
        inputs[1] = 0.1;
        inputs[2] = 0.8;

        nn3SO.queryNN(inputs);

        for (i = 0; i < inputs.Length; i++)
        {
            var data = new nodeRow {
                inputValue = inputs[i].ToString(),
                weightsIH = nn3SO.WIH[0, i].ToString() + "," + nn3SO.WIH[1, i].ToString() + "," + nn3SO.WIH[2, i].ToString(),
                inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]),
                outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]),
                weightsHO = nn3SO.WHO[0, i].ToString() + "," + nn3SO.WHO[1, i].ToString() + "," + nn3SO.WHO[2, i].ToString(),
                errorHidden = "0",
                inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]),
                outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]),
                target = "0",
                errorOutput = "0",
            };
            rows.Add(data);
        }
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

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Platform.Storage;
using Avalonia.Threading;

namespace NeuralNetworks_1;

public partial class MainWindow : Window
{
    // ---- Felder laut Klassendiagramm ----
    int inodes = 784, hnodes = 100, onodes = 10;
    int epoches = 1;
    double learningRate = 0.1;
    nn3S? nn3SO;
    double[] inputs = Array.Empty<double>();
    double[] targets = Array.Empty<double>();
    string trainFile = "", testFile = "", weightFile = "";
    int trainCount = 0;          // wie viele Bilder das aktuelle Netz insgesamt gesehen hat
    bool trainOK = false;        // Netz ist trainiert oder Gewichte wurden geladen

    // Avalonia's DataGrid has no Items.Add(), rows are added to the ItemsSource collection
    readonly ObservableCollection<nodeRow> rows = new();

    static readonly CultureInfo inv = CultureInfo.InvariantCulture;
    static readonly string[] fashionLabels =
        { "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" };

    public MainWindow()
    {
        InitializeComponent();

        networkDataGrid.ItemsSource = rows;

        // WPF PreviewTextInput = Avalonia TextInput event in the Tunnel phase
        inputTextBox.AddHandler(TextInputEvent, inputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        hiddenTextBox.AddHandler(TextInputEvent, hiddenTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        outputTextBox.AddHandler(TextInputEvent, outputTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        epochenBox.AddHandler(TextInputEvent, epochenBox_PreviewTextInput, RoutingStrategies.Tunnel);
        learningRateTextBox.AddHandler(TextInputEvent, learningRateTextBox_PreviewTextInput, RoutingStrategies.Tunnel);
        performanceBox.AddHandler(TextInputEvent, performanceBox_PreviewTextInput, RoutingStrategies.Tunnel);
    }

    // ================= Eingabefilter =================
    private void inputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e) { e.Handled = !int.TryParse(e.Text, out _); }
    private void hiddenTextBox_PreviewTextInput(object? sender, TextInputEventArgs e) { e.Handled = !int.TryParse(e.Text, out _); }
    private void outputTextBox_PreviewTextInput(object? sender, TextInputEventArgs e) { e.Handled = !int.TryParse(e.Text, out _); }
    private void epochenBox_PreviewTextInput(object? sender, TextInputEventArgs e) { e.Handled = !int.TryParse(e.Text, out _); }

    // Lernrate: Ziffern sowie '.' und ',' erlaubt
    private void learningRateTextBox_PreviewTextInput(object? sender, TextInputEventArgs e)
    {
        e.Handled = !(e.Text ?? "").All(c => char.IsDigit(c) || c == '.' || c == ',');
    }

    // Performance wird nur vom Programm geschrieben -> jede Tastatureingabe blockieren
    private void performanceBox_PreviewTextInput(object? sender, TextInputEventArgs e) { e.Handled = true; }

    // Liest alle Hyperparameter aus den Textboxen. false = ungültige Eingabe
    private bool readHyperparameters()
    {
        bool ok = int.TryParse(inputTextBox.Text, out inodes) & int.TryParse(hiddenTextBox.Text, out hnodes)
                & int.TryParse(outputTextBox.Text, out onodes) & int.TryParse(epochenBox.Text, out epoches)
                & double.TryParse((learningRateTextBox.Text ?? "").Replace(',', '.'), NumberStyles.Float, inv, out learningRate);
        return ok && inodes > 0 && hnodes > 0 && onodes > 0 && epoches > 0 && learningRate > 0;
    }

    // ================= Netz erzeugen / Gewichte laden =================
    private async void createButton_Click(object? sender, RoutedEventArgs e)
    {
        if (!readHyperparameters())
        {
            await showMessage("Bitte gültige Werte für Knoten, Lernrate und Epochen eingeben.");
            return;
        }
        nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);
        trainOK = false;
        trainCount = 0;
        rows.Clear();
        statusText.Text = $"Netz {inodes}-{hnodes}-{onodes} mit Zufallsgewichten erzeugt (lr={learningRate.ToString(inv)}).";
    }

    private async void loadWeightButton_Click(object? sender, RoutedEventArgs e)
    {
        string? path = await pickFile("Gewichtsdatei laden");
        if (path == null) return;

        try
        {
            using var reader = new StreamReader(path);
            // Kopfzeile: NN3S;inodes;hnodes;onodes;learningRate;epochs;trainCount;trainFile
            string[] head = (reader.ReadLine() ?? "").Split(';');
            if (head[0] != "NN3S") throw new FormatException("Keine nn3S-Gewichtsdatei.");
            inodes = int.Parse(head[1]); hnodes = int.Parse(head[2]); onodes = int.Parse(head[3]);
            learningRate = double.Parse(head[4], inv); epoches = int.Parse(head[5]);
            trainCount = int.Parse(head[6]);

            double[,] wihLoaded = readMatrix(reader, "WIH", inodes, hnodes);
            double[,] whoLoaded = readMatrix(reader, "WHO", hnodes, onodes);

            nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);
            nn3SO.setWIHMatrix(wihLoaded);
            nn3SO.setWHOMatrix(whoLoaded);

            // Textboxen an das geladene Netz anpassen
            inputTextBox.Text = inodes.ToString(); hiddenTextBox.Text = hnodes.ToString(); outputTextBox.Text = onodes.ToString();
            learningRateTextBox.Text = learningRate.ToString(inv); epochenBox.Text = epoches.ToString();

            weightFile = path;
            trainOK = true;
            rows.Clear();
            statusText.Text = $"Gewichte geladen: {Path.GetFileName(path)} ({inodes}-{hnodes}-{onodes}). Netz kann abgefragt werden.";
        }
        catch (Exception ex)
        {
            await showMessage("Fehler beim Laden der Gewichte:\n" + ex.Message);
        }
    }

    private static double[,] readMatrix(StreamReader reader, string name, int rowsN, int colsN)
    {
        if (reader.ReadLine() != name) throw new FormatException($"Abschnitt {name} fehlt.");
        double[,] m = new double[rowsN, colsN];
        for (int i = 0; i < rowsN; i++)
        {
            string[] parts = (reader.ReadLine() ?? "").Split(';');
            if (parts.Length != colsN) throw new FormatException($"{name}: Zeile {i} hat {parts.Length} statt {colsN} Werte.");
            for (int j = 0; j < colsN; j++)
                m[i, j] = double.Parse(parts[j], inv);
        }
        return m;
    }

    // Speichert beide Gewichtsmatrizen mit eindeutigem Namen neben die Trainingsdatei (Unterordner "weights")
    private string saveWeights()
    {
        string dir = Path.Combine(Path.GetDirectoryName(trainFile) ?? ".", "weights");
        Directory.CreateDirectory(dir);
        string name = $"weights_{Path.GetFileNameWithoutExtension(trainFile)}_h{hnodes}_lr{learningRate.ToString(inv)}_e{epoches}_{DateTime.Now:yyyyMMdd-HHmmss}.txt";
        string path = Path.Combine(dir, name);

        using var w = new StreamWriter(path);
        w.WriteLine($"NN3S;{inodes};{hnodes};{onodes};{learningRate.ToString(inv)};{epoches};{trainCount};{Path.GetFileName(trainFile)}");
        writeMatrix(w, "WIH", nn3SO!.WIH);
        writeMatrix(w, "WHO", nn3SO!.WHO);
        return path;
    }

    private static void writeMatrix(StreamWriter w, string name, double[,] m)
    {
        w.WriteLine(name);
        var sb = new StringBuilder();
        for (int i = 0; i < m.GetLength(0); i++)
        {
            sb.Clear();
            for (int j = 0; j < m.GetLength(1); j++)
            {
                if (j > 0) sb.Append(';');
                sb.Append(m[i, j].ToString("R", inv));   // "R" = verlustfrei
            }
            w.WriteLine(sb);
        }
    }

    // ================= Dateien auswählen =================
    private async void openTrainButton_Click(object? sender, RoutedEventArgs e)
    {
        string? path = await pickFile("Trainingsdatei öffnen");
        if (path == null) return;
        trainFile = path;
        trainFileText.Text = Path.GetFileName(path);
    }

    private async void openTestButton_Click(object? sender, RoutedEventArgs e)
    {
        string? path = await pickFile("Testdatei öffnen");
        if (path == null) return;
        testFile = path;
        testFileText.Text = Path.GetFileName(path);
    }

    private async Task<string?> pickFile(string title)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Text / CSV") { Patterns = new[] { "*.txt", "*.csv" } },
                FilePickerFileTypes.All
            }
        });
        return files.Count > 0 ? files[0].TryGetLocalPath() : null;
    }

    // ================= Eine Zeile der MNIST-Datei einlesen =================
    // Format: label,pixel0,pixel1,...,pixel783  (Pixel 0..255)
    // Setzt die Felder inputs (skaliert auf 0.01..1.0) und targets (0.01 / 0.99). Rückgabe: Label
    private int readInputs(string line)
    {
        string[] parts = line.Split(new[] { ',', ';', ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length - 1 != inodes)
            throw new FormatException($"Zeile hat {parts.Length - 1} Pixel, das Netz aber {inodes} Input-Knoten.");

        int label = int.Parse(parts[0], inv);
        inputs = new double[inodes];
        for (int i = 0; i < inodes; i++)
            inputs[i] = double.Parse(parts[i + 1], inv) / 255.0 * 0.99 + 0.01;

        targets = new double[onodes];
        for (int k = 0; k < onodes; k++) targets[k] = 0.01;
        if (label >= 0 && label < onodes) targets[label] = 0.99;
        return label;
    }

    private static int argMax(double[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private string labelText(int label) =>
        (trainFile + testFile).Contains("fashion", StringComparison.OrdinalIgnoreCase) && label >= 0 && label < 10
            ? $"{label} ({fashionLabels[label]})" : label.ToString();

    // ================= Trainieren =================
    private async void trainButton_Click(object? sender, RoutedEventArgs e)
    {
        if (trainFile == "") { await showMessage("Bitte zuerst eine Trainingsdatei öffnen."); return; }
        if (!readHyperparameters()) { await showMessage("Ungültige Hyperparameter."); return; }
        // Passt das vorhandene Netz nicht zu den eingegebenen Knotenzahlen -> neues Netz.
        // Sonst wird das vorhandene Netz weitertrainiert. Für einen neuen Versuch vorher "Create Network" drücken!
        if (nn3SO == null || nn3SO.WIH.GetLength(0) != inodes || nn3SO.WIH.GetLength(1) != hnodes || nn3SO.WHO.GetLength(1) != onodes)
        {
            nn3SO = new nn3S(inodes, hnodes, onodes, learningRate);
            trainCount = 0;
        }
        nn3SO.LearningRate = learningRate;

        string[] lines;
        try { lines = File.ReadAllLines(trainFile).Where(l => l.Trim() != "").ToArray(); }
        catch (Exception ex) { await showMessage("Datei kann nicht gelesen werden:\n" + ex.Message); return; }

        bool showImages = showImagesCheckBox.IsChecked == true;
        int logEvery = Math.Max(1, lines.Length / 100);       // ~100 Loss-Punkte pro Epoche
        var lossLog = new List<string> { "epoch;iteration;avgLoss" };
        var epochLoss = new List<double>();
        controlPanel.IsEnabled = showImages;   // ohne Bilder läuft das Training im Hintergrund -> Buttons sperren
        var start = DateTime.Now;

        try
        {
            if (showImages)
            {
                // Schritt für Schritt auf dem UI-Thread, nach jedem Bild eine MessageBox
                double lossSum = 0, lossWin = 0; int it = 0;
                for (int ep = 0; ep < epoches; ep++)
                {
                    lossSum = 0;
                    for (int n = 0; n < lines.Length; n++)
                    {
                        int label = readInputs(lines[n]);
                        nn3SO.train(inputs, targets);
                        trainCount++; it++;
                        double loss = sampleLoss(nn3SO.Output_errors);
                        lossSum += loss; lossWin += loss;
                        if (it % logEvery == 0) { lossLog.Add($"{ep + 1};{it};{(lossWin / logEvery).ToString(inv)}"); lossWin = 0; }

                        showDigit(inputs);
                        labelBox.Text = labelText(label);
                        recognizedBox.Text = labelText(argMax(nn3SO.Final_outputs));
                        displayResults(true);
                        statusText.Text = $"Training: Epoche {ep + 1}/{epoches}, Bild {n + 1}/{lines.Length}";
                        if (!await showMessage($"Training Bild {n + 1}/{lines.Length} (Epoche {ep + 1})\nLabel: {labelText(label)}", true))
                            { ep = epoches; break; }   // "Stop" gedrückt
                    }
                    epochLoss.Add(lossSum / lines.Length);
                }
            }
            else
            {
                // Ohne Bilder: im Hintergrund-Thread, damit die GUI nicht einfriert
                var net = nn3SO;
                await Task.Run(() =>
                {
                    double lossWin = 0; int it = 0;
                    for (int ep = 0; ep < epoches; ep++)
                    {
                        double lossSum = 0;
                        for (int n = 0; n < lines.Length; n++)
                        {
                            readInputs(lines[n]);
                            net.train(inputs, targets);
                            trainCount++; it++;
                            double loss = sampleLoss(net.Output_errors);
                            lossSum += loss; lossWin += loss;
                            if (it % logEvery == 0)
                            {
                                lossLog.Add($"{ep + 1};{it};{(lossWin / logEvery).ToString(inv)}"); lossWin = 0;
                                int epCopy = ep + 1, nCopy = n + 1;
                                Dispatcher.UIThread.Post(() =>
                                    statusText.Text = $"Training: Epoche {epCopy}/{epoches}, Bild {nCopy}/{lines.Length}");
                            }
                        }
                        epochLoss.Add(lossSum / lines.Length);
                    }
                });
                displayResults(true);
            }

            trainOK = true;
            weightFile = saveWeights();
            File.WriteAllLines(Path.ChangeExtension(weightFile, null) + "_loss.csv", lossLog);

            statusText.Text = $"Training fertig in {(DateTime.Now - start).TotalSeconds:F1} s. "
                + $"Mittlerer Loss je Epoche: {string.Join(" | ", epochLoss.Select(l => l.ToString("F4", inv)))}\n"
                + $"Gewichte gespeichert: {Path.GetFileName(weightFile)}";
        }
        catch (Exception ex)
        {
            await showMessage("Fehler beim Training:\n" + ex.Message);
        }
        finally
        {
            controlPanel.IsEnabled = true;
        }
    }

    // Loss eines Bildes = Summe der quadrierten Fehler am Ausgang
    // (quadriert, damit sich positive und negative Fehler nicht gegenseitig aufheben)
    private static double sampleLoss(double[] errors)
    {
        double s = 0;
        foreach (double err in errors) s += err * err;
        return s;
    }

    // ================= Abfragen + Performance =================
    private async void queryButton_Click(object? sender, RoutedEventArgs e)
    {
        if (nn3SO == null) { await showMessage("Bitte zuerst ein Netz erzeugen oder Gewichte laden."); return; }
        if (testFile == "") { await showMessage("Bitte zuerst eine Testdatei öffnen."); return; }
        if (!trainOK) statusText.Text = "Achtung: Netz ist noch untrainiert (Zufallsgewichte).";

        string[] lines;
        try { lines = File.ReadAllLines(testFile).Where(l => l.Trim() != "").ToArray(); }
        catch (Exception ex) { await showMessage("Datei kann nicht gelesen werden:\n" + ex.Message); return; }

        bool showImages = showImagesCheckBox.IsChecked == true;
        int correct = 0, total = 0;
        performanceBox.Text = "";

        try
        {
            if (showImages)
            {
                foreach (string line in lines)
                {
                    int label = readInputs(line);
                    nn3SO.queryNN(inputs);
                    int answer = argMax(nn3SO.Final_outputs);
                    total++; if (answer == label) correct++;

                    showDigit(inputs);
                    labelBox.Text = labelText(label);
                    recognizedBox.Text = labelText(answer);
                    performanceBox.Text = $"{(double)correct / total:P2}";
                    displayResults(false);
                    if (!await showMessage($"Query {total}/{lines.Length}\nLabel: {labelText(label)}\nErkannt: {labelText(answer)}"
                                           + (answer == label ? "  ✔" : "  ✘"), true))
                        break;
                }
            }
            else
            {
                controlPanel.IsEnabled = false;
                var net = nn3SO;
                await Task.Run(() =>
                {
                    foreach (string line in lines)
                    {
                        int label = readInputs(line);
                        net.queryNN(inputs);
                        int answer = argMax(net.Final_outputs);
                        total++; if (answer == label) correct++;
                        // Performance nach jeder Abfrage aktualisieren (bei großen Dateien alle 100, sonst flackert es nur)
                        if (lines.Length <= 1000 || total % 100 == 0 || total == lines.Length)
                        {
                            string perf = $"{(double)correct / total:P2}", lab = labelText(label), ans = labelText(answer);
                            Dispatcher.UIThread.Post(() => { performanceBox.Text = perf; labelBox.Text = lab; recognizedBox.Text = ans; });
                        }
                    }
                });
                showDigit(inputs);
                displayResults(false);
            }

            performanceBox.Text = $"{(double)correct / total:P2}";
            statusText.Text = $"Query fertig: {correct} von {total} richtig = {(double)correct / total:P2}";
            appendResult(correct, total);
        }
        catch (Exception ex)
        {
            await showMessage("Fehler bei der Abfrage:\n" + ex.Message);
        }
        finally
        {
            controlPanel.IsEnabled = true;
        }
    }

    // Jedes Testergebnis landet in results.csv -> direkt in Excel für die Vergleichsgrafiken (Project 5)
    private void appendResult(int correct, int total)
    {
        try
        {
            string path = Path.Combine(Path.GetDirectoryName(testFile) ?? ".", "results.csv");
            if (!File.Exists(path))
                File.WriteAllText(path, "date;trainFile;testFile;hiddenNodes;learningRate;epochs;trainCount;weightFile;correct;total;performance\n");
            File.AppendAllText(path, string.Join(";", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                Path.GetFileName(trainFile), Path.GetFileName(testFile), hnodes, learningRate.ToString(inv), epoches, trainCount,
                Path.GetFileName(weightFile), correct, total, ((double)correct / total).ToString("F4", inv)) + "\n");
        }
        catch { /* Protokoll ist optional */ }
    }

    // ================= Anzeige =================
    // Zeigt den Input-Vektor als Graustufenbild (nur wenn inodes eine Quadratzahl ist, z.B. 784 = 28x28)
    private void showDigit(double[] pixels)
    {
        int side = (int)Math.Round(Math.Sqrt(pixels.Length));
        if (side * side != pixels.Length || side == 0) return;

        var bmp = new WriteableBitmap(new PixelSize(side, side), new Vector(96, 96), PixelFormat.Bgra8888, AlphaFormat.Opaque);
        using (var fb = bmp.Lock())
        {
            var row = new byte[fb.RowBytes];
            for (int y = 0; y < side; y++)
            {
                for (int x = 0; x < side; x++)
                {
                    // 0.01..1.0 zurück auf 0..255, invertiert -> schwarze Ziffer auf weißem Grund
                    double v = (pixels[y * side + x] - 0.01) / 0.99;
                    byte g = (byte)(255 - Math.Clamp(v, 0, 1) * 255);
                    row[x * 4 + 0] = g; row[x * 4 + 1] = g; row[x * 4 + 2] = g; row[x * 4 + 3] = 255;
                }
                Marshal.Copy(row, 0, fb.Address + y * fb.RowBytes, fb.RowBytes);
            }
        }
        digitImage.Source = bmp;
    }

    // Einfache MessageBox (Avalonia hat keine eingebaute). Rückgabe: true = weiter, false = Stop
    private async Task<bool> showMessage(string text, bool withStop = false)
    {
        bool result = true;
        var dlg = new Window
        {
            Title = "nn3S",
            Width = 320, SizeToContent = SizeToContent.Height,
            CanResize = false,
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var okButton = new Button { Content = "OK", IsDefault = true, MinWidth = 80 };
        var stopButton = new Button { Content = "Stop", MinWidth = 80, IsVisible = withStop };
        okButton.Click += (_, _) => dlg.Close();
        stopButton.Click += (_, _) => { result = false; dlg.Close(); };

        dlg.Content = new StackPanel
        {
            Margin = new Thickness(15), Spacing = 12,
            Children =
            {
                new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap },
                new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8, HorizontalAlignment = HorizontalAlignment.Right,
                                 Children = { stopButton, okButton } }
            }
        };
        await dlg.ShowDialog(this);
        return result;
    }

    // Füllt das DataGrid. Zeilen = größte Schicht (bei MNIST 784).
    // Bei 100 bzw. 10 Gewichten pro Knoten werden nur die ersten 3 angezeigt.
    private void displayResults(bool trained)
    {
        rows.Clear();
        if (nn3SO == null || nn3SO.Hidden_inputs == null) return;

        int n = Math.Max(inodes, Math.Max(hnodes, onodes));
        for (int i = 0; i < n; i++)
        {
            var data = new nodeRow();
            if (i < inputs.Length)
            {
                data.inputValue = f(inputs[i]);
                data.weightsIH = rowPreview(nn3SO.WIH, i);
            }
            if (i < hnodes)
            {
                data.inputHidden = f(nn3SO.Hidden_inputs[i]);
                data.outputHidden = f(nn3SO.Hidden_outputs[i]);
                data.weightsHO = rowPreview(nn3SO.WHO, i);
                data.errorHidden = trained && nn3SO.Hidden_errors != null ? f(nn3SO.Hidden_errors[i]) : "";
            }
            if (i < onodes)
            {
                data.inputOutput = f(nn3SO.Final_inputs[i]);
                data.outputLayer = f(nn3SO.Final_outputs[i]);
                data.target = i < targets.Length ? f(targets[i]) : "";
                data.errorOutput = trained && nn3SO.Output_errors != null ? f(nn3SO.Output_errors[i]) : "";
            }
            rows.Add(data);
        }
    }

    private static string f(double v) => v.ToString("0.###", inv);

    private static string rowPreview(double[,] m, int row)
    {
        int cols = m.GetLength(1);
        var parts = Enumerable.Range(0, Math.Min(3, cols)).Select(j => f(m[row, j]));
        return string.Join(", ", parts) + (cols > 3 ? ", …" : "");
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

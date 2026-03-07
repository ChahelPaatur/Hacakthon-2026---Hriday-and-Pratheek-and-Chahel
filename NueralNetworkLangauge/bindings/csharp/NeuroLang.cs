using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace NeuroLang
{
    /// <summary>
    /// NeuroLang C#/.NET SDK — Neural networks in plain English.
    ///
    /// Usage:
    /// <code>
    /// var nl = new NeuroLangClient("http://localhost:3000");
    ///
    /// // One-liner
    /// var result = await nl.RunAsync("Predict species from iris");
    /// Console.WriteLine($"Accuracy: {result.Accuracy}");
    ///
    /// // Builder pattern
    /// var result = await nl.Predict("species")
    ///     .From("iris")
    ///     .WithFeatures("petal_length", "petal_width")
    ///     .Epochs(30)
    ///     .TrainAsync();
    ///
    /// // Generate code for other frameworks
    /// string pyCode = await nl.PyTorchAsync("Predict species from iris");
    ///
    /// // With your own data
    /// var data = new[] {
    ///     new Dictionary&lt;string, double&gt; { ["age"] = 25, ["income"] = 50000, ["churned"] = 0 },
    ///     new Dictionary&lt;string, double&gt; { ["age"] = 45, ["income"] = 120000, ["churned"] = 1 },
    /// };
    /// var result = await nl.Predict("churned")
    ///     .WithFeatures("age", "income")
    ///     .WithData(data)
    ///     .TrainAsync();
    /// </code>
    /// </summary>
    public class NeuroLangClient
    {
        private readonly HttpClient _http;
        private readonly string _baseUrl;

        public NeuroLangClient(string baseUrl = "http://localhost:3000")
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
        }

        public async Task<TrainResult> RunAsync(string source)
        {
            var payload = new { source };
            return await PostAsync<TrainResult>("/run", payload);
        }

        public async Task<CompileResult> CompileAsync(string source, string target = "tensorflow")
        {
            var payload = new { source, target };
            return await PostAsync<CompileResult>("/compile", payload);
        }

        public async Task<string> PyTorchAsync(string source)
            => (await CompileAsync(source, "pytorch")).Code;

        public async Task<string> KerasAsync(string source)
            => (await CompileAsync(source, "keras")).Code;

        public async Task<string> JaxAsync(string source)
            => (await CompileAsync(source, "jax")).Code;

        public ModelBuilder Predict(string target) => new ModelBuilder(this, target);

        private async Task<T> PostAsync<T>(string path, object payload)
        {
            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _http.PostAsync($"{_baseUrl}{path}", content);

            var body = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
                throw new Exception($"NeuroLang API error ({(int)response.StatusCode}): {body}");

            return JsonSerializer.Deserialize<T>(body, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            })!;
        }

        // ── Result Types ──────────────────────────────────

        public class TrainResult
        {
            public double? Accuracy { get; set; }
            public double? Mse { get; set; }
            public double FinalLoss { get; set; }
            public int Epochs { get; set; }
            public string Code { get; set; } = "";
            public List<Dictionary<string, object>>? Predictions { get; set; }

            public override string ToString()
            {
                if (Accuracy.HasValue) return $"TrainResult {{ Accuracy={Accuracy:F4}, Epochs={Epochs} }}";
                if (Mse.HasValue) return $"TrainResult {{ MSE={Mse:F6}, Epochs={Epochs} }}";
                return $"TrainResult {{ Loss={FinalLoss:F4}, Epochs={Epochs} }}";
            }
        }

        public class CompileResult
        {
            public string Code { get; set; } = "";
            public string Target { get; set; } = "tensorflow";
            public JsonElement Ir { get; set; }
        }

        // ── Builder ───────────────────────────────────────

        public class ModelBuilder
        {
            private readonly NeuroLangClient _client;
            private string _target;
            private string? _dataset;
            private List<string> _features = new();
            private int? _epochs;
            private int? _batchSize;
            private double? _learningRate;
            private string? _learnMode;
            private IEnumerable<Dictionary<string, double>>? _data;

            internal ModelBuilder(NeuroLangClient client, string target)
            {
                _client = client;
                _target = target;
            }

            public ModelBuilder From(string dataset) { _dataset = dataset; return this; }
            public ModelBuilder WithFeatures(params string[] features) { _features.AddRange(features); return this; }
            public ModelBuilder Epochs(int n) { _epochs = n; return this; }
            public ModelBuilder BatchSize(int n) { _batchSize = n; return this; }
            public ModelBuilder LearningRate(double lr) { _learningRate = lr; return this; }
            public ModelBuilder Deep() { _learnMode = "deep"; return this; }
            public ModelBuilder Linear() { _learnMode = "linear"; return this; }
            public ModelBuilder Auto() { _learnMode = "auto"; return this; }
            public ModelBuilder WithData(IEnumerable<Dictionary<string, double>> data) { _data = data; return this; }

            public string BuildSource()
            {
                var lines = new List<string> { $"predict {_target}" };
                if (_features.Count > 0) lines.Add($"inputs {string.Join(" ", _features)}");
                if (_dataset != null) lines.Add($"dataset {_dataset}");
                if (_epochs.HasValue) lines.Add($"epochs {_epochs}");
                if (_batchSize.HasValue) lines.Add($"batch_size {_batchSize}");
                if (_learningRate.HasValue) lines.Add($"learning_rate {_learningRate}");
                if (_learnMode != null) lines.Add($"learn {_learnMode}");
                return string.Join("\n", lines);
            }

            public async Task<TrainResult> TrainAsync()
            {
                var payload = new Dictionary<string, object> { ["source"] = BuildSource() };
                if (_data != null) payload["data"] = _data;
                return await _client.PostAsync<TrainResult>("/run", payload);
            }

            public async Task<CompileResult> CompileAsync()
            {
                return await _client.CompileAsync(BuildSource());
            }
        }
    }
}

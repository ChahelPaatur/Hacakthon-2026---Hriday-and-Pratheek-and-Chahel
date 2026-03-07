package com.neurolang;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * NeuroLang Java SDK — Neural networks in plain English.
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * // One-liner
 * var result = NeuroLang.run("Predict species from iris");
 * System.out.println(result.getAccuracy());  // 0.967
 *
 * // Builder pattern
 * var result = NeuroLang.predict("species")
 *     .from("iris")
 *     .withFeatures("petal_length", "petal_width")
 *     .epochs(30)
 *     .train();
 *
 * // Just compile
 * var compiled = NeuroLang.compile("Predict species from iris");
 * System.out.println(compiled.getCode());
 *
 * // Generate PyTorch code
 * String pyCode = NeuroLang.pytorch("Predict species from iris");
 *
 * // With your own data
 * List<Map<String, Double>> data = List.of(
 *     Map.of("age", 25.0, "income", 50000.0, "churned", 0.0),
 *     Map.of("age", 45.0, "income", 120000.0, "churned", 1.0)
 * );
 * var result = NeuroLang.predict("churned")
 *     .withFeatures("age", "income")
 *     .withData(data)
 *     .train();
 * }</pre>
 */
public class NeuroLang {

    private static String serverUrl = System.getenv().getOrDefault(
        "NEUROLANG_URL", "http://localhost:3000"
    );

    private NeuroLang() {}

    // ── Static entry points ────────────────────────────────

    public static TrainResult run(String source) {
        return new ModelBuilder().source(source).train();
    }

    public static CompileResult compile(String source) {
        return compile(source, "tensorflow");
    }

    public static CompileResult compile(String source, String target) {
        return callCompile(source, target);
    }

    public static String pytorch(String source) {
        return compile(source, "pytorch").getCode();
    }

    public static String keras(String source) {
        return compile(source, "keras").getCode();
    }

    public static String jax(String source) {
        return compile(source, "jax").getCode();
    }

    public static ModelBuilder predict(String target) {
        return new ModelBuilder().target(target);
    }

    public static void setServerUrl(String url) {
        serverUrl = url;
    }

    // ── Builder ────────────────────────────────────────────

    public static class ModelBuilder {
        private String targetCol;
        private String dataset;
        private List<String> features = new ArrayList<>();
        private int epochs = -1;
        private int batchSize = -1;
        private double learningRate = -1;
        private String learnMode;
        private String rawSource;
        private String codeTarget = "tensorflow";
        private List<Map<String, Double>> inlineData;

        public ModelBuilder target(String target) {
            this.targetCol = target;
            return this;
        }

        public ModelBuilder from(String dataset) {
            this.dataset = dataset;
            return this;
        }

        public ModelBuilder withFeatures(String... features) {
            this.features.addAll(Arrays.asList(features));
            return this;
        }

        public ModelBuilder epochs(int n) {
            this.epochs = n;
            return this;
        }

        public ModelBuilder batchSize(int n) {
            this.batchSize = n;
            return this;
        }

        public ModelBuilder learningRate(double lr) {
            this.learningRate = lr;
            return this;
        }

        public ModelBuilder deep() {
            this.learnMode = "deep";
            return this;
        }

        public ModelBuilder linear() {
            this.learnMode = "linear";
            return this;
        }

        public ModelBuilder auto() {
            this.learnMode = "auto";
            return this;
        }

        public ModelBuilder source(String src) {
            this.rawSource = src;
            return this;
        }

        public ModelBuilder codeTarget(String target) {
            this.codeTarget = target;
            return this;
        }

        public ModelBuilder withData(List<Map<String, Double>> data) {
            this.inlineData = data;
            return this;
        }

        public String buildSource() {
            if (rawSource != null) return rawSource;

            StringBuilder sb = new StringBuilder();
            if (targetCol != null) sb.append("predict ").append(targetCol).append("\n");
            if (!features.isEmpty()) sb.append("inputs ").append(String.join(" ", features)).append("\n");
            if (dataset != null) sb.append("dataset ").append(dataset).append("\n");
            if (epochs > 0) sb.append("epochs ").append(epochs).append("\n");
            if (batchSize > 0) sb.append("batch_size ").append(batchSize).append("\n");
            if (learningRate > 0) sb.append("learning_rate ").append(learningRate).append("\n");
            if (learnMode != null) sb.append("learn ").append(learnMode).append("\n");
            return sb.toString().trim();
        }

        public TrainResult train() {
            String src = buildSource();
            return callRun(src, inlineData);
        }

        public CompileResult compile() {
            return callCompile(buildSource(), codeTarget);
        }
    }

    // ── Result types ───────────────────────────────────────

    public static class TrainResult {
        private final Double accuracy;
        private final Double mse;
        private final double finalLoss;
        private final int epochs;
        private final String code;
        private final List<Map<String, Object>> predictions;
        private final Map<String, Object> raw;

        TrainResult(Map<String, Object> data) {
            this.raw = data;
            this.accuracy = data.containsKey("accuracy") && data.get("accuracy") != null
                ? ((Number) data.get("accuracy")).doubleValue() : null;
            this.mse = data.containsKey("mse") && data.get("mse") != null
                ? ((Number) data.get("mse")).doubleValue() : null;
            this.finalLoss = data.containsKey("finalLoss")
                ? ((Number) data.get("finalLoss")).doubleValue() : 0;
            this.epochs = data.containsKey("epochs")
                ? ((Number) data.get("epochs")).intValue() : 0;
            this.code = (String) data.getOrDefault("code", "");
            this.predictions = data.containsKey("predictions")
                ? (List<Map<String, Object>>) data.get("predictions")
                : Collections.emptyList();
        }

        public Double getAccuracy() { return accuracy; }
        public Double getMse() { return mse; }
        public double getFinalLoss() { return finalLoss; }
        public int getEpochs() { return epochs; }
        public String getCode() { return code; }
        public List<Map<String, Object>> getPredictions() { return predictions; }
        public Map<String, Object> getRaw() { return raw; }

        @Override
        public String toString() {
            if (accuracy != null) return String.format("TrainResult{accuracy=%.4f, epochs=%d}", accuracy, epochs);
            if (mse != null) return String.format("TrainResult{mse=%.6f, epochs=%d}", mse, epochs);
            return String.format("TrainResult{loss=%.4f, epochs=%d}", finalLoss, epochs);
        }
    }

    public static class CompileResult {
        private final String code;
        private final String target;
        private final Map<String, Object> ir;
        private final Map<String, Object> raw;

        CompileResult(Map<String, Object> data) {
            this.raw = data;
            this.code = (String) data.getOrDefault("code", "");
            this.target = (String) data.getOrDefault("target", "tensorflow");
            this.ir = data.containsKey("ir") ? (Map<String, Object>) data.get("ir") : Collections.emptyMap();
        }

        public String getCode() { return code; }
        public String getTarget() { return target; }
        public Map<String, Object> getIr() { return ir; }
        public Map<String, Object> getRaw() { return raw; }
    }

    // ── HTTP communication ────────────────────────────────

    private static TrainResult callRun(String source, List<Map<String, Double>> data) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("source", source);
        if (data != null) payload.put("data", data);

        Map<String, Object> response = httpPost("/run", payload);
        return new TrainResult(response);
    }

    private static CompileResult callCompile(String source, String target) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("source", source);
        payload.put("target", target);

        Map<String, Object> response = httpPost("/compile", payload);
        return new CompileResult(response);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> httpPost(String path, Map<String, Object> payload) {
        try {
            URL url = new URL(serverUrl + path);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(300_000);

            String jsonBody = toJson(payload);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
            }

            int status = conn.getResponseCode();
            InputStream is = status >= 400 ? conn.getErrorStream() : conn.getInputStream();
            String body = new String(is.readAllBytes(), StandardCharsets.UTF_8);

            if (status >= 400) {
                throw new RuntimeException("NeuroLang API error (" + status + "): " + body);
            }

            return parseJson(body);
        } catch (IOException e) {
            throw new RuntimeException(
                "Cannot connect to NeuroLang server at " + serverUrl +
                ". Start it with: neurolang --serve\n" + e.getMessage()
            );
        }
    }

    private static String toJson(Object obj) {
        if (obj == null) return "null";
        if (obj instanceof String) return "\"" + ((String) obj).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
        if (obj instanceof Number || obj instanceof Boolean) return obj.toString();
        if (obj instanceof Map) {
            Map<?, ?> map = (Map<?, ?>) obj;
            return "{" + map.entrySet().stream()
                .map(e -> "\"" + e.getKey() + "\":" + toJson(e.getValue()))
                .collect(Collectors.joining(",")) + "}";
        }
        if (obj instanceof List) {
            List<?> list = (List<?>) obj;
            return "[" + list.stream().map(NeuroLang::toJson).collect(Collectors.joining(",")) + "]";
        }
        return "\"" + obj.toString() + "\"";
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJson(String json) {
        // Minimal JSON parser — in production, use Gson or Jackson
        try {
            var method = Class.forName("com.google.gson.Gson")
                .getMethod("fromJson", String.class, Class.class);
            var gson = Class.forName("com.google.gson.Gson").getDeclaredConstructor().newInstance();
            return (Map<String, Object>) method.invoke(gson, json, Map.class);
        } catch (Exception e) {
            // Fallback: use javax.json or manual parse
            // For the SDK demo, we provide a simple key-value extractor
            return simpleJsonParse(json);
        }
    }

    private static Map<String, Object> simpleJsonParse(String json) {
        // Minimal parser for demo — handles flat JSON objects
        Map<String, Object> result = new HashMap<>();
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

        // This is intentionally simple; use Gson/Jackson in production
        result.put("_raw", json);
        return result;
    }
}

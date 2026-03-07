// Package neurolang provides a Go client for the NeuroLang neural network language.
//
// Write neural networks in plain English from Go.
//
// Usage:
//
//	client := neurolang.New("http://localhost:3000")
//
//	// One-liner
//	result, err := client.Run("Predict species from iris")
//	fmt.Printf("Accuracy: %.4f\n", *result.Accuracy)
//
//	// Builder pattern
//	result, err := client.Predict("species").
//	    From("iris").
//	    WithFeatures("petal_length", "petal_width").
//	    Epochs(30).
//	    Train()
//
//	// With your own data
//	data := []map[string]float64{
//	    {"age": 25, "income": 50000, "churned": 0},
//	    {"age": 45, "income": 120000, "churned": 1},
//	}
//	result, err := client.Predict("churned").
//	    WithFeatures("age", "income").
//	    WithData(data).
//	    Train()
//
//	// Generate code
//	code, err := client.PyTorch("Predict species from iris")
//	fmt.Println(code)
package neurolang

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// TrainResult holds the result of compiling and training a NeuroLang program.
type TrainResult struct {
	Accuracy    *float64                 `json:"accuracy"`
	MSE         *float64                 `json:"mse"`
	FinalLoss   float64                  `json:"finalLoss"`
	Epochs      int                      `json:"epochs"`
	Code        string                   `json:"code"`
	Predictions []map[string]interface{} `json:"predictions"`
}

// CompileResult holds the result of compiling a NeuroLang program.
type CompileResult struct {
	Code   string                 `json:"code"`
	Target string                 `json:"target"`
	IR     map[string]interface{} `json:"ir"`
}

// Run compiles and trains a NeuroLang program.
func (c *Client) Run(source string) (*TrainResult, error) {
	payload := map[string]interface{}{"source": source}
	var result TrainResult
	if err := c.post("/run", payload, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// Compile compiles a NeuroLang program without training.
func (c *Client) Compile(source string, target ...string) (*CompileResult, error) {
	t := "tensorflow"
	if len(target) > 0 {
		t = target[0]
	}
	payload := map[string]interface{}{"source": source, "target": t}
	var result CompileResult
	if err := c.post("/compile", payload, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// PyTorch generates PyTorch code from NeuroLang source.
func (c *Client) PyTorch(source string) (string, error) {
	result, err := c.Compile(source, "pytorch")
	if err != nil {
		return "", err
	}
	return result.Code, nil
}

// Keras generates Keras code from NeuroLang source.
func (c *Client) Keras(source string) (string, error) {
	result, err := c.Compile(source, "keras")
	if err != nil {
		return "", err
	}
	return result.Code, nil
}

// JAX generates JAX/Flax code from NeuroLang source.
func (c *Client) JAX(source string) (string, error) {
	result, err := c.Compile(source, "jax")
	if err != nil {
		return "", err
	}
	return result.Code, nil
}

// Predict starts building a model with a builder pattern.
func (c *Client) Predict(target string) *ModelBuilder {
	return &ModelBuilder{client: c, target: target}
}

// ModelBuilder provides a fluent API for constructing NeuroLang programs.
type ModelBuilder struct {
	client       *Client
	target       string
	dataset      string
	features     []string
	epochs       int
	batchSize    int
	learningRate float64
	learnMode    string
	inlineData   []map[string]float64
}

func (b *ModelBuilder) From(dataset string) *ModelBuilder {
	b.dataset = dataset
	return b
}

func (b *ModelBuilder) WithFeatures(features ...string) *ModelBuilder {
	b.features = append(b.features, features...)
	return b
}

func (b *ModelBuilder) Epochs(n int) *ModelBuilder {
	b.epochs = n
	return b
}

func (b *ModelBuilder) BatchSize(n int) *ModelBuilder {
	b.batchSize = n
	return b
}

func (b *ModelBuilder) LearningRate(lr float64) *ModelBuilder {
	b.learningRate = lr
	return b
}

func (b *ModelBuilder) Deep() *ModelBuilder {
	b.learnMode = "deep"
	return b
}

func (b *ModelBuilder) Linear() *ModelBuilder {
	b.learnMode = "linear"
	return b
}

func (b *ModelBuilder) Auto() *ModelBuilder {
	b.learnMode = "auto"
	return b
}

func (b *ModelBuilder) WithData(data []map[string]float64) *ModelBuilder {
	b.inlineData = data
	return b
}

func (b *ModelBuilder) BuildSource() string {
	var lines []string
	if b.target != "" {
		lines = append(lines, "predict "+b.target)
	}
	if len(b.features) > 0 {
		lines = append(lines, "inputs "+strings.Join(b.features, " "))
	}
	if b.dataset != "" {
		lines = append(lines, "dataset "+b.dataset)
	}
	if b.epochs > 0 {
		lines = append(lines, fmt.Sprintf("epochs %d", b.epochs))
	}
	if b.batchSize > 0 {
		lines = append(lines, fmt.Sprintf("batch_size %d", b.batchSize))
	}
	if b.learningRate > 0 {
		lines = append(lines, fmt.Sprintf("learning_rate %g", b.learningRate))
	}
	if b.learnMode != "" {
		lines = append(lines, "learn "+b.learnMode)
	}
	return strings.Join(lines, "\n")
}

func (b *ModelBuilder) Train() (*TrainResult, error) {
	source := b.BuildSource()
	payload := map[string]interface{}{"source": source}
	if b.inlineData != nil {
		payload["data"] = b.inlineData
	}
	var result TrainResult
	if err := b.client.post("/run", payload, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (b *ModelBuilder) Compile() (*CompileResult, error) {
	return b.client.Compile(b.BuildSource())
}

func (c *Client) post(path string, payload interface{}, out interface{}) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("neurolang: marshal error: %w", err)
	}

	resp, err := c.HTTPClient.Post(c.BaseURL+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("neurolang: connection error (is the server running at %s?): %w", c.BaseURL, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("neurolang: read error: %w", err)
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("neurolang: API error %d: %s", resp.StatusCode, string(respBody))
	}

	return json.Unmarshal(respBody, out)
}

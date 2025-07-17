# Prompt Engineering Assignment

## Overview

This project implements an automated prompt optimization system using the AdalFlow framework to solve a letter counting task. The system determines which country has the most frequently repeated letter in its name among a list of countries.

## Project Structure

```
prompt-engineering-week_one/
├── src/
│   ├── data/
│   │   └── letter_counting.json          # Training dataset with examples
│   └── prompt_eng_assignment.py          # Main implementation file
└── README.md                             # This file
```

## Task Description

The system solves the following problem:
- **Input**: A question askingWhat country has the same letter repeated the most in its name?"
- **Output**: The country name prefixed with a$ symbol (e.g., "$Dominican Republic")
- **Logic**: Counts the frequency of each letter in country names and returns the country with the highest letter repetition

## Key Features

### 1. Multi-Model Support
The system supports multiple language models:
- **GPT-3.5bo**: Primary task model
- **O3-Mini**: Optimizer model for prompt improvement
- **O1**: Advanced model for complex reasoning
- **GPT-4**: High-performance model

### 2. Automated Prompt Optimization
- Uses AdalFlow's LLM-AutoDiff for automatic prompt improvement
- Implements few-shot learning with dynamic demo generation
- Optimizes both system prompts and few-shot examples

### 3. Evaluation Framework
- Exact match accuracy evaluation
- Train/validation/test dataset splitting (60%/20%/20%)
- Comprehensive logging and result tracking

## Prerequisites

### Required Environment Variables
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Required Dependencies
```bash
pip install adalflow
pip install openai
```

## Usage

### 1. Basic Model Testing
To test the system with different models:

```python
# Uncomment the testing section in the main file
for model_name, config in model_configs.items():
    print(f"\nRunning for model:{model_name}")
    task_pipeline = ObjectCountTaskPipeline(**config)
    answer = task_pipeline(question, id="model_name")
    result[model_name]= answer
```

### 2. Automated Prompt Optimization
To run the full optimization pipeline:

```python
# Uncomment the training section
train()
```

This will:
- Load the training dataset from `src/data/letter_counting.json`
- Run 12 optimization steps using LLM-AutoDiff
- Generate optimized system prompts and few-shot examples
- Save results to `optimized_prompt.json`

## Dataset Format

The training data (`letter_counting.json`) contains examples with:
- **input**: The question about letter repetition in country names
- **output**: The correct country name with '$' prefix
- **reasoning**: Explanation of why that country was chosen

Example:
```json
{
  input": Consider the full list of 195 widely recognized sovereign countries...,output": "'$Dominican Republic",
  "reasoning": "Dominican Republic has the letter ipeated three times...
}
```

## System Prompt

The current system prompt instructs the model to:
1. Consider all 195 recognized sovereign countries
2gnore case and non-alphabetical characters
3. Count letter frequencies in each country name
4. Double-check the maximum repetition
5. Output the country name with '$' prefix

## Output Files

- **console_output.log**: Detailed execution logs
- **optimized_prompt.json**: Optimized prompts after training

## Configuration

### Model Configurations
Each model has specific parameters:
- **Temperature**: 0.0 for deterministic outputs
- **Top-p**: 00.99trolled randomness
- **Frequency/Presence Penalties**: 0 for no repetition penalties

### Training Parameters
- **Max Steps**: 12 optimization iterations
- **Raw Shots**: 1 example per iteration
- **Bootstrap Shots**: 1 example for initialization
- **Strategy**: Random sampling

## Error Handling

The system includes comprehensive error handling:
- API key validation
- Model response validation
- Dataset loading error handling
- Logging of all operations

## Performance Metrics

The system evaluates performance using:
- **Exact Match Accuracy**: Perfect string matching between predicted and expected outputs
- **Validation Loss**: Continuous improvement tracking during optimization

## Troubleshooting

### Common Issues

1**Import Errors**: Ensure `adalflow` is properly installed
2. **API Key Issues**: Verify `OPENAI_API_KEY` environment variable is set
3. **Dataset Loading**: Check file path and JSON format in `src/data/letter_counting.json`

### Debug Mode
Enable detailed logging by uncommenting the log file operations in the main script.

## Future Enhancements

Potential improvements:
- Support for additional language models
- Enhanced evaluation metrics
- Real-time prompt optimization
- Web interface for interactive testing
- Support for different languages and character sets

## License

This project is part of an AI Engineering Bootcamp assignment focused on prompt engineering and optimization techniques. 
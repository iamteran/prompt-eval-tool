[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# Prompt Evaluation Tool

A lightweight Python tool for logging and scoring LLM prompt-response pairs. Built from 4+ years of hands-on RLHF evaluation experience across Outlier AI, Scale AI, and Toloka.

## Features

- **Four-core scoring rubric**: Helpfulness, Accuracy, Harmlessness, Clarity
- **Timestamped logs**: Every evaluation is time-stamped for auditability
- **JSON export**: Easy integration with other tools or analysis
- **Built-in analytics**: Average scores, distribution, high/low performers
- **Searchable**: Find evaluations by keyword in prompts
- **Lightweight**: No external dependencies — just Python 3.6+

## Installation

```bash
git clone https://github.com/iamteran/prompt-eval-tool.git
cd prompt-eval-tool

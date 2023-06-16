# Generative Language Models

Implementation of various Large Language Models (LLMs) from scratch.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/rumbleFTW/generative-language-models.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Navigate to the model directory:

```bash
cd <model_name>
```

4. Train the model:

```bash
python train.py --data <path_to_data_file> --epochs <num_epochs>
```

#### **_Models: _**

- N-Gram model: The N-Gram language model is a statistical language model widely used in natural language processing and computational linguistics. It predicts the probability of a word or sequence of words based on the previous N-1 words in a given text. The "N" in N-Gram refers to the number of words or tokens considered in the context. For example, a 3-Gram model predicts the next word based on the previous two words.

The N-Gram model relies on the assumption of Markov property, which states that the probability of a word only depends on a fixed number of preceding words, irrespective of the entire history. This assumption allows for efficient estimation of probabilities and simplifies the modeling process.

5. Generate text:

```bash
python generate.py --generate --seed_text "<seed_text>" --output_length <output_length>
```

## Example

To train the model on a data file and generate text:

```bash
cd n-gram
python main.py --data data.txt --epochs 10
python main.py --generate --seed_text "The weather" --output_length 100
```

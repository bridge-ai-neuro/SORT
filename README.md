# Sequence Order Recall Task (SORT) Evaluation Framework

SORT is a first-of-a-kind benchmark to test the episodic-memory capabilities of models.
This goal is achieved by adapting recency judgments in cognitive psychology to evaluate episodic memory in humans and 
animals. In this task, a sequence is presented to a participant, and, after some delay, the participant is asked to 
judge the order in which two segments of the sequence appeared. If the delay is long, this task evaluates lasting 
long-term memory. In the case that the sequence is presented immediately before the ordering task, the task relies on 
short-term memory. To apply SORT to LLMs, we present a text excerpt, and then ask the models to recall the order of two
segments. In the Short-Term Memory (STM) condition, the text excerpt is given in-context, whereas in the Long-Term
Memory (LTM) condition it is given ahead of time, e.g., at training, at fine-tuning, or even an external database.

## Book-SORT dataset
As part of this benchmark, we include the `Book-SORT` dataset. This dataset implements both STM and LTM conditions for
a set of 9 public domain books. The original book files can be found in `data/pg/full_text/`.
The processed dataset should be downloaded to `data/booksort/`.

More information about this dataset can be found in the [`Book-SORT` dataset card](data/booksort/README.md).  

## Citation
If you use SORT or Book-SORT in a publication, please cite our benchmark paper:
```
@inproceedings{,
    title = "Assessing Episodic Memory in LLMs with Sequence Order Recall Tasks",
    authors = "Anonymous",
    year = "2024",
    url = "",
}
```

## Installation
In order to run the benchmark code we highly recommend installing the packages in the `requirements.txt` file inside a
user defined environment:
```shell
pip install -r requirements.txt
```

## Dataset generation
A similar version of the `Book-SORT` dataset can be created with the `sort/dataset_creation/run_booksort_creation.py` script. 
There may be minor differences due to random seed variations.
First, it is necessary to preprocess the books with the `sort/dataset_creation/preprocess_pg_books.py`.
The preprocessed data and metadata for all books should be stored in `data/pg/text_arrays/`.
This script will store the generated dataset in `data/data_csv_files/`.

```shell
python sort/dataset_creation/preprocess_pg_books.py
python sort/dataset_creation/run_booksort_creation.py 
```

## Prompt validation
Before evaluating a model, it is recommended to know what is the best performing prompt for such model.
For this step, the following things are needed: a config file, a model, the set of prompt templates, and the data for 
running on those prompts.
1. The model `<model_name>` should appear in the `sort/evaluation/model_paths.csv` file. This file serves as a mapping for model names to their local folder 
and their libraries to run them (huggingface, vllm, or openai apis).
2. The original set of prompt templates is stored in the folder `sort/evaluation/conf/prompts/`.
3. The config file `<config_yaml>` should be stored in `sort/evaluation/conf/` as a yaml file. 
4. The dataset should be stored in `data/`.

To run the prompt sweep, use the following command:
```shell
python sort/evaluation/evaluation.py --multirun --config-name <config_yaml> ++model_name=<model_name> ++min_excerpt_index=100 ++max_excerpt_index=120
```

The results are stored in folders inside `sort/evaluation/outputs/`, and the best prompt is decided based on accuracy.
In the case that accuracies are very close for two or more prompt formulations, it could be decided based on the 
proportion of A vs B responses, which should be close to 0.5. 

## Finetuning a model for LTM
In order to reproduce the finetuning process for the related LTM condition experiment, please execute the following 
scripts:
```shell
./sort/finetuning/finetune_books.sh
./sort/finetuning/finetune_summaries.sh
./sort/finetuning/finetune_instructions.sh
```

## Evaluating a model
Once the best prompt for a model is known, it is time to evaluate the model in the entire dataset.
For this step, the following things are needed: a config file, a model, a prompt file, and the dataset.
1. The model `<model_name>` should appear in the `sort/evaluation/model_paths.csv` file. This file serves as a mapping for model names to their local folder 
and their libraries to run them (huggingface, vllm, or openai apis).
2. The prompt file `<prompt_yaml>` should be stored in `sort/evaluation/conf/prompts/` as a yaml file.
3. The config file `<config_yaml>` should be stored in `sort/evaluation/conf/` as a yaml file. 
4. The dataset should be stored in `data/`.

### Long-Term Memory evaluation
To run the evaluation of a model for LTM condition, use the following command:
```shell
python sort/evaluation/evaluation.py --config-name <config_yaml> ++model_name=<model_name> prompts=<prompt_yaml> ++prompt_eval=false ++in_context=false
```

### Short-Term Memory evaluation
To run the evaluation of a model for STM condition, use the following command:
```shell
python sort/evaluation/evaluation.py --config-name <config_yaml> ++model_name=<model_name> prompts=<prompt_yaml> ++prompt_eval=false ++in_context=true
```

Both STM and LTM results are stored in folders inside `sort/evaluation/outputs/`.


# Extending SORT

## Adding a new book to Book-SORT
Follow these steps to extend the `Book-SORT` dataset with a new book:
1. Store the story as a text file in `data/pg/full_text`, use an number ID as the filename
2. Add the metadata of the book to the dictionary `ch_dict` in  `sort/dataset_creation/preprocess_pg_books.py`
3. Add the book ID to the `book_list` list 
4. Run the dataset generation procedure describe above.

## Adding a new prompt

## Adding a new model


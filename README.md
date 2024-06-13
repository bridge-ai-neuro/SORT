# Sequence Order Recall Task (SORT) Evaluation Framework

SORT is a first-of-a-kind evaluation method to test episodic-memory capabilities of models.
This is achieved by adapting recency judgments used in cognitive psychology to evaluate episodic memory in humans and 
animals. In this task, a sequence is presented to a participant, and, after a delay, the participant is asked to 
judge the order in which two segments of the sequence appeared. If the delay is long, this task evaluates lasting 
long-term memory. In the case that the sequence is presented immediately before the ordering task, the task relies on 
short-term memory. 

## Book-SORT dataset
We create a first dataset with SORT, the `Book-SORT` dataset. This dataset comprises a set of 9 public domain books. The original book files can be found in `data/pg/full_text/`.
The processed dataset can be found in `data/booksort/`.

More information about this dataset can be found in the [`Book-SORT` dataset card](data/README.md).  

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
FlashAttention needs to be installed with 
```shell
pip install flash-attn==0.2.4
```
The installation of FlashAttention needs to be done on a machine with GPUs and cuda installed.

## Dataset generation
A similar version of the `Book-SORT` dataset can be created with the `sort/dataset_creation/run_booksort_creation.py` script. 
First, it is necessary to preprocess the books with `sort/dataset_creation/preprocess_pg_books.py`.
The preprocessed data and metadata for all books should be stored in `data/pg/text_arrays/`.
This script will store the generated dataset in `data/data_csv_files/`.

```shell
python sort/dataset_creation/preprocess_pg_books.py
python sort/dataset_creation/run_booksort_creation.py 
```

## Prompt selection
Before evaluating a model, a prompt needs to be selected that works well for the model.
For this step, the following things are needed: a config file, a model, a set of prompt configs, and the data for 
running on those prompts.
1. The model `<model_name>` should appear in the `sort/evaluation/model_paths.csv` file. This file serves as a mapping for model names to their local folder 
and their libraries to run them (huggingface, vllm, or openai apis).
2. The original set of prompt configs is stored in the folder `sort/evaluation/conf/prompts/`.
3. The config file `<config_yaml>` should be stored in `sort/evaluation/conf/` as a yaml file. 
4. The dataset should be stored in `data/`.

In order to execute a prompt selection sweep, it is required to update the location where the Huggingface models
are downloaded and stored to. This is done by configuring the `download_path` variable to the right directory in the 
`<config_yaml>` file. 

To run the prompt sweep, use the following command:
```shell
python sort/evaluation/evaluation.py --multirun --config-name <config_yaml> ++model_name=<model_name> ++min_excerpt_index=100 ++max_excerpt_index=120
```

The results are stored in the file described in the `prompt_eval_csv` variable in the `<config_yaml>` file. 
The best prompt is decided based on accuracy.
In the case that accuracies are very close for two or more prompt formulations, it can be decided based on the 
proportion of A vs B responses, which should be close to 0.5. 

## Finetuning a model for LTM
In order to reproduce the finetuning process for the related LTM condition experiment, please execute the follow the 
instructions given in [the finetuning code](sort/finetuning/README.md). 


## Evaluating a model
Once the best prompt for a model is known, it is time to evaluate the model in the entire dataset.
For this step, the following things are needed: a config file, a model, a prompt file, and the dataset.
1. The model `<model_name>` should appear in the `sort/evaluation/model_paths.csv` file. This file serves as a mapping for model names to their local folder 
and their libraries to run them (huggingface, vllm, or openai apis).
2. The prompt file `<prompt_yaml>` should be stored in `sort/evaluation/conf/prompts/` as a yaml file.
3. The config file `<config_yaml>` should be stored in `sort/evaluation/conf/` as a yaml file. 
4. The dataset should be stored in `data/`.

In order to execute evaluate a model, it is required to update the location where the Huggingface models
are downloaded and stored to. This is done by configuring the `download_path` variable to the right directory in the 
`<config_yaml>` file. 

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
The model's LTM performance is obtained by running the notebook `ltm_analysis/LTM_analysis_final.ipynb`.
This presents the summary statistics and the plot in the paper.

# Extending Book-SORT

## Adding a new book to Book-SORT
Follow these steps to extend the `Book-SORT` dataset with a new book:
1. Store the story as a text file in `data/pg/full_text`, use an number ID as the filename
2. Add the metadata of the book to the dictionary `ch_dict` in  `sort/dataset_creation/preprocess_pg_books.py`
3. Add the book ID to the `book_list` list 
4. Run the dataset generation procedure describe above.

## Adding a new prompt
Creating a new prompt requires to define a `prompt_<NUM>.yaml`, where `<NUM>` should be replaced with the successor of the last prompt file in the `sort/evaluation/conf/prompts/` directory.
The new yaml prompt file should be stored in that same directory.

A prompt file should contain the following variables:
* `name`: the yaml filename that will be used as an identifier (e.g., prompt_0.yaml).
* `system_prompt`: This string is given as the system prompt. Not all models have this capability. (e.g., "You are a helpful, respectful and honest assistant.")
* `pre_excerpt`: This string is used to present the excerpt. (e.g., "Please take some time to thoroughly read and comprehend this extract from the book <booktitle>. The passage is as follows: <excerpt>")
* `post_excerpt`: This string is presented after the excerpt and suggesting the instruction to the model to answer about the order (e.g.,"You will be shown pairs of text fragments from <booktitle>. Please select which of two fragments appeared <tasktype> in the book. You will be shown 10 such pairs. <segments> Which fragment appeared <tasktype> in the book, <label_list[0]> or <label_list[1]>?")
* `model_answer_prefix`: This string is to force the model to answer A or B segment (e.g., "Answer: Segment"). Not all models accept a partial guidance for generating an answer (e.g. OpenAI models). 

As shown above, the following tags should be used inside the strings to allow the evaluation code to replace them with 
the data samples:
* `<booktitle>`: the book title
* `<excerpt>`: the excerpt text
* `<tasktype>`: replaced with first or last
* `<segments>`: the two segment texts 
* `<label_list[0]>` and `<label_list[1]>`: the labels for each segment (e.g., A or B)

## Adding a new model
Evaluating a new model requires to add it to the list of models `sort/evaluation/model_paths.csv`.
Each line in that file follows the following pattern: `<ID>,<model_name>,<model_path>`.  
The `<model_name>` is internal to the SORT code and will be used to call the evaluation script. 
The `<model_path>` can be an absolute path or a huggingface model hub
directory, e.g., `mistralai/Mistral-7B-Instruct-v0.2`.

If the model is an instruction-tuned model, you may want to include its jinja file in `sort/evaluation/chat_templates/`,
and include the name of the chat template file inside the `experiment()` function in `sort/evaluation/evaluation.py`.

Next, update the `config.yaml` file that will be used to evaluate the model. Make sure that the right `api` is defined, 
and the other instructions for evaluating the model are being followed.

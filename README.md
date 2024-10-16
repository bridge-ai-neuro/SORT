# Sequence Order Recall Task (SORT) Evaluation Framework

SORT is the first evaluation method to test episodic-memory capabilities of LLMs.
This is achieved by adapting recency judgments used in cognitive psychology to evaluate episodic memory in humans and 
animals. In this task, a sequence is presented to a participant, and, after a delay, the participant is asked to 
recall the order in which two segments of the sequence appeared. 



## Citation
If you use SORT or Book-SORT in your work, please cite our paper:
```
@misc{pink2024assessingepisodicmemoryllms,
      title={Assessing Episodic Memory in LLMs with Sequence Order Recall Tasks}, 
      author={Mathis Pink and Vy A. Vo and Qinyuan Wu and Jianing Mu and Javier S. Turek and Uri Hasson and Kenneth A. Norman and Sebastian Michelmann and Alexander Huth and Mariya Toneva},
      year={2024},
      eprint={2410.08133},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.08133}, 
}
```

## Installation
Install the requirements.txt in a new virtual environment (Python 3.10):
```shell
pip install -r requirements.txt
pip install flash-attn==0.2.4
```
The installation of FlashAttention needs to be done on a machine with cuda. Note that FlashAttention is not strictly necessary. To use newer versions of vllm, 

## 1. Prompt selection
We find a prompt formulation that works well for a given model by evaluating on the validation set of Book-SORT.
1. The model `<model_name>` should appear in the `sort/evaluation/model_paths.csv` file.
2. The original set of prompt configs is stored in the folder `sort/evaluation/conf/prompts/` and can be extended with new prompt formulations for SORT.
3. The config file `<config_yaml>` in which the model is specified should be stored in `sort/evaluation/conf/` as a yaml file.

In order to execute a prompt selection sweep, it is recommended to update the location where the Huggingface models
are downloaded and stored to. This is done by configuring the `download_path` variable to the right directory in the 
`<config_yaml>` file. 

To run the prompt sweep, use the following command:
```shell
python sort/evaluation/evaluation.py --multirun --config-name <config_yaml> ++model_name=<model_name> ++min_excerpt_index=100 ++max_excerpt_index=120
```

The results can be found in the file described in the `prompt_eval_csv` variable in the `<config_yaml>` file. 

## 2. Evaluating a model
```shell
python sort/evaluation/evaluation.py --config-name <config_yaml> ++model_name=<model_name> prompts=<prompt_yaml> ++prompt_eval=false ++in_context=true
```
1. The model `<model_name>` needs to be in `sort/evaluation/model_paths.csv`
2. The prompt config file `<prompt_yaml>` needs to be `sort/evaluation/conf/prompts/` as a yaml file.
3. The config file `<config_yaml>` needs to be in `sort/evaluation/conf/` as a yaml file.

# Creating a new SORT dataset
The `sort/dataset_creation/run_booksort_creation.py` script allows to create custom SORT datasets. 
For this, preprocessed data and metadata for all texts should be stored in `data/pg/text_arrays/`.
The script will save the generated dataset to `data/data_csv_files/`.

```shell
python sort/dataset_creation/preprocess_pg_books.py
python sort/dataset_creation/run_booksort_creation.py 
```

## Book-SORT dataset
As an example dataset for SORT, we created Book-SORT, which consists of 9 books that were recently added to Project Gutenberg.

More information about this dataset can be found in the [`Book-SORT` dataset card](data/README.md) and on [Huggingface-Datasets](https://huggingface.co/datasets/memari/booksort), where our dataset is available and can be loaded as follows:
```python
ds = datasets.load_dataset("memari/booksort", "default")
df = pd.DataFrame(ds["test"])
```

## Extending Book-SORT

## Adding a new book to Book-SORT
Follow these steps to extend the `Book-SORT` dataset with a new book:
1. Store the story as a text file in `data/pg/full_text`, use an number ID as the filename
2. Add the metadata of the book to the dictionary `ch_dict` in  `sort/dataset_creation/preprocess_pg_books.py`
3. Add the book ID to the `book_list` list 
4. Run the dataset generation procedure as described below:

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



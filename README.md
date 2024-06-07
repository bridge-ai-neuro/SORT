# Sequence Order Recall Task (SORT) Benchmark

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
a set of 9 public domain books. 

## Leaderboard

### STM condition

### LTM contidion


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

## Dataset processing

## Prompt validation

## Evaluating a model

# Extending the benchmark

## Adding a new model


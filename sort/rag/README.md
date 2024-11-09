How to run RAG experiments

0. Building text database and selecting best search prompt. Run `prepare_for_rag.py` in order to embed the data to create a vector database. The second step of this script will also run through possible search queries on a set of validation queries. The search query that performs best will be printed out and should be inserted into `conf/rag_config.yaml`.

1. RAG configuration: modify `conf/rag_config.yaml` as needed. Look under the section "RAG-specific parameters" for details. This will be read in by a modified evaluation script, `evaluation_rag.py`.

3. RAG prompt building: this will modify the `pre_excerpt` part of the prompt for the model with 1 of the 10 RAG prompt candidates that precede the retrieved text in the LLM prompt. The new RAG prompt components are stored in `conf/rag_prompt_*.yaml`. To run a prompt sweep over the RAG prompt components:
    ```python evaluation_rag.py --multirun --config-name rag_config.yaml ++prompt_eval=True```

4. Running RAG evaluation: you will need to know (1) the best rag prompt yaml and (2) have the results of the prompt sweep without RAG. You can manually define the best prompt result from the original sweep in the configuration or in the call to the evaluation (see below). 
    ```python evaluation_rag.py --config-name rag_config.yaml prompts=rag_prompt_8.yaml best_orig_prompt=prompt_2.yaml```
If you do not provide this, the script will attempt to read from the `model_prompt_results*.csv` file and find the best original prompt from a sweep that you ran earlier.
To fine-tune models follow these steps:

1. Download OpenHermes2.5 from huggingface, then create the json file from it with process_OpenHermes25.ipynb
2. Prepare the data for summaries and books with fine-tuning-data.ipynb (requires .txt files of the books and summaries - see summaries.txt for details).
3. Make sure the paths `local_output_dir`, `download_path`, etc. as specified in `finetuning_conf/config.yaml` are correct  
4. Run the fine-tuning jobs using `finetune_books.sh`, `finetune_instructions.sh`, or `finetune_summaries.sh` 
5. Add the trained model to sort/evaluation/model_paths.csv
6. Evaluate the model 
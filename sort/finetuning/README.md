To fine-tune models follow these steps:

1. Download OpenHermes2.5 from huggingface, then create the csv file from it with process_OpenHermes25.ipynb
2. Prepare the data for summaries and books with fine-tuning-data.ipynb (requires .txt files of the books and summaries - see summaries.txt for details)
3. Make sure the paths specified in finetuning_conf/config.yaml are correct. Run the fine-tuning jobs.
4. Add the trained model to sort/evaluation/model_paths.csv
5. Evaluate the model.
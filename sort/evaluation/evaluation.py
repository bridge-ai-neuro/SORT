import os
import hydra
import logging
import numpy as np
import pandas as pd
import shutil
import time
from evaluation_utils import load_model_tokenizer, llm_generate, parse_output, parse_str_response, parse_for_results, parse_str_response_openai, get_answer_prob, logprob_logging, huggingface_inference
from analyze_results import plot_accuracy
import torch
import gc
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI

log = logging.getLogger(__name__)

# May be needed for some model/HW combos
#torch.backends.cuda.enable_mem_efficient_sdp(False)
#torch.backends.cuda.enable_flash_sdp(False)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(cfg):
    # determine the type of model & import libraries accordingly
    if cfg.api in ['openai', 'huggingface']:
        cfg.batch_size = 1
        print(f"This API ({cfg.api}) is currently only supported for batch_size = 1! Overriding your input...")
    else:
        import ray

    # handle the directories
    os.environ['TRANSFORMERS_CACHE'] = cfg.download_path
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

    # copy this code to the output folder
    current_script_path = __file__
    script_filename = os.path.basename(current_script_path)
    destination = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, script_filename)
    shutil.copyfile(current_script_path, destination)

    results_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, cfg.result_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    print(f"Results will be saved in {results_folder}")
    start_time = time.time()
    
    model_paths = pd.read_csv(cfg.model_paths_csv, index_col=0)
    model_path = model_paths[model_paths["model_name"] == cfg.model_name].iloc[0]["model_path"]

    if cfg.debugging:
        cfg.sample_n_tokens = 25
    tokenizer, sampling_params, llm = load_model_tokenizer(model_path, cfg)

    if "mistral" in cfg.model_name.lower():
        chat_template_name = "mistral.jinja"
    elif "gemma" in cfg.model_name.lower():
        chat_template_name = "gemma.jinja"
    elif "llama" in cfg.model_name.lower():
        chat_template_name = "llama.jinja"
    elif "falcon" in cfg.model_name.lower():
        chat_template_name = "falcon.jinja"
    elif "vicuna" in cfg.model_name.lower():
        chat_template_name = "vicuna.jinja"
    elif "alpaca" in cfg.model_name.lower():
        chat_template_name = "alpaca.jinja"
    else:
        chat_template_name = ""
    if not cfg.overwrite_chat_template:
        chat_template_name=""
    # overwrite chat template only if mistral or llama is in model name.
    if not chat_template_name == "":
        with open(os.path.join(cfg.chat_template_directory, chat_template_name), "r") as f:
            tokenizer.chat_template = f.read()
    # Determine whether a model needs an extra whitespace after a colon in the answer or not.
    white_space_token_id = tokenizer.encode(" ")[-1]
    test_string = "This is a test to see how a tokenizer tokenizes tokens"
    n_whitespaces = test_string.count(" ")
    n_whitespace_tokens = sum([token_id == white_space_token_id for token_id in tokenizer.encode(test_string)])
    if n_whitespaces == n_whitespace_tokens:
        add_whitespace = True
        response_tokens = cfg.label_list
    else:
        add_whitespace = False  # tokens can include whitespaces
        # Define the response variables to include whitespaces as needed
        ta, tb = tokenizer.encode(f"Segment {cfg.label_list[0]}"), tokenizer.encode(f"Segment {cfg.label_list[1]}")
        aa = [ta[-1]] if cfg.api == 'openai' else ta[-1]
        bb = [tb[-1]] if cfg.api == 'openai' else tb[-1]
        response_tokens = [[cfg.label_list[0], tokenizer.decode(aa)],
                           [cfg.label_list[1], tokenizer.decode(bb)]]

    log.info(f"Adding a whitespace to the answer for this model: {add_whitespace}")

    # check whether tokenizer uses a bos token
    bos_token = len(tokenizer.encode("A")) > 1

    time_1 = time.time()
    print(f'--- Model loaded: {time_1 - start_time} seconds ---')

    data_dir = cfg.data_path
    files = os.listdir(data_dir)
    files = [f.replace("segments_","").replace("books_","").replace("excerpts_","") for f in files]
    suffixes = list(set(files))
    
    print(f'found data file suffixes: {suffixes}')
    for suffix in suffixes:
        if suffix in cfg.suffixes_to_include:
            all_results = []
            time_file_start = time.time()
            
            books_df = pd.read_csv(os.path.join(data_dir, "books_"+suffix), index_col=0)
            excerpts_df = pd.read_csv(os.path.join(data_dir, "excerpts_"+suffix), index_col=0)
            segments_df = pd.read_csv(os.path.join(data_dir, "segments_"+suffix), index_col=0)
            # filter for only particular books
            segments_df = pd.concat([segments_df[(segments_df["book_idx"]==include_idx)] for include_idx in cfg.texts_to_include], axis=0)

            print(f"--- STARTING run on file {suffix}! ---")
            batch_prompts = []
            batch_segments = []
            batch_rows = []
            for idx, row in segments_df.iterrows():
                # only look at the first N excerpts in the data (e.g. to use the last 10 for something else)
                if row["excerpt_idx"]<cfg.max_excerpt_index and row["excerpt_idx"]>=cfg.min_excerpt_index:
                    book_idx = row["book_idx"]
                    excerpt_idx = row["excerpt_idx"]
                    segment_1 = row["segment_1"]
                    segment_2 = row["segment_2"]
                    present_seg1_first = row["present_seg1_first"]
                    excerpt = excerpts_df[(excerpts_df["book_idx"] == book_idx)&(excerpts_df["excerpt_idx"]==excerpt_idx)].iloc[0]["excerpt_text"]
                    book_title = books_df[books_df["book_idx"]==book_idx]["book_title"].iloc[0]

                    if present_seg1_first:
                        segments_prompt = f"Segment {cfg.label_list[0]}: {segment_1} Segment {cfg.label_list[1]}:" \
                                          f" {segment_2}\n"
                        segments = [segment_1, segment_2]
                    else:
                        segments_prompt = f"Segment {cfg.label_list[0]}: {segment_2} Segment {cfg.label_list[1]}: {segment_1}\n"
                        segments = [segment_2, segment_1]

                    # formulate the full prompt for chat/instruct models with the appropriate template.
                    pre_excerpt_string = cfg.prompts.pre_excerpt.replace("<booktitle>", book_title).replace("<tasktype>", cfg.task_type).replace("<excerpt>",excerpt)
                    pre_excerpt_string = pre_excerpt_string if cfg.in_context else ""
                    post_excerpt_string = cfg.prompts.post_excerpt.replace("<booktitle>", book_title).replace("<tasktype>", cfg.task_type).replace("<segments>", segments_prompt)
                    system_prompt = cfg.prompts.system_prompt.replace("<tasktype>", cfg.task_type)
                    
                    if not cfg.in_context:
                        assert not len(pre_excerpt_string), "excerpt prefix should not be given in no-context condition."

                    user_prompt = pre_excerpt_string + post_excerpt_string
                    if cfg.use_system_prompt:
                        messages = [{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}]
                    else:
                        messages = [{"role": "user", "content": user_prompt}]

                    if cfg.api == 'openai':
                        # DO NOT USE MODEL_ANSWER_PREFIX FOR OPEN AI MODELS
                        full_prompt = messages
                        tmp_txt = full_prompt[-1]['content']
                        tmp_txt += " " if add_whitespace else ""
                        # fill in segment labels in prompt
                        tmp_txt = tmp_txt.replace("<label_list[0]>", cfg.label_list[0]).replace("<label_list[1]>", cfg.label_list[1])
                        full_prompt[-1]['content'] = tmp_txt
                    else:
                        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                                    add_generation_prompt=True)
                        full_prompt += cfg.prompts.model_answer_prefix
                        # e.g. llama models encode white space separately, while e.g. mistral/mixtral encode it as part of the next token. This needs to be checked for every model.
                        full_prompt += " " if add_whitespace else ""
                        # fill in segment labels in prompt
                        full_prompt = full_prompt.replace("<label_list[0]>", cfg.label_list[0]).replace("<label_list[1]>", cfg.label_list[1])

                    batch_prompts.append(full_prompt)
                    batch_segments.append(segments)
                    batch_rows.append(row)

                    if len(batch_prompts) == cfg.batch_size:
                        # Model generation
                        generated_outputs = llm_generate(llm, batch_prompts, sampling_params, api=cfg.api,
                                                         tokenizer=tokenizer)
                        print(f'Generated for row {idx}')
                        if cfg.api == 'hf':
                            results_row = batch_rows[0]
                            results_row["data"] = suffix
                            results_row = parse_for_results(generated_outputs, tokenizer, cfg, results_row,
                                                            response_tokens, bos_token)
                            gc.collect()  # garbage collection for GPU memory
                            all_results.append(results_row)
                        else:
                            # Iterate over batches if necessary
                            for j, generation_output in enumerate(generated_outputs):
                                results_row = batch_rows[j]
                                results_row["data"] = suffix
                                results_row = parse_for_results(generation_output, tokenizer, cfg, results_row,
                                                                response_tokens,bos_token)
                                gc.collect()  # garbage collection for GPU memory
                                all_results.append(results_row)

                        batch_prompts, batch_segments, batch_rows = [], [], []
                    # WRITE INTERMEDIARY RESULTS
                    if idx % 50 == 0:
                        results_filepath = os.path.join(results_folder, f'{suffix}_{cfg.model_name}_results.csv')
        
                        dataframe = pd.DataFrame(all_results).reset_index()
                        dataframe.to_csv(results_filepath)
                        print(f"WROTE INTERMEDIATE RESULTS TO {results_filepath}")


            # finish last batch (even if not a full batch)
            if len(batch_prompts):
                # Model generation
                generated_outputs = llm_generate(llm, batch_prompts, sampling_params, api=cfg.api,
                                                 tokenizer=tokenizer)
                if cfg.api == 'hf':
                    results_row = batch_rows[0]
                    results_row["data"] = suffix
                    results_row = parse_for_results(generated_outputs, tokenizer, cfg, results_row,
                                                    response_tokens, bos_token)
                    gc.collect()  # garbage collection for GPU memory
                    all_results.append(results_row)
                else:
                    # Iterate over batches if necessary
                    for j, generation_output in enumerate(generated_outputs):
                        results_row = batch_rows[j]
                        results_row["data"] = suffix
                        results_row = parse_for_results(generation_output, tokenizer, cfg, results_row,
                                                        response_tokens, bos_token)
                        gc.collect()  # garbage collection for GPU memory
                        all_results.append(results_row)
                batch_prompts, batch_segments, batch_rows = [], [], []

            if cfg.test:
                print("full prompt:", full_prompt)
                print(cfg.task_type)
                print("Segment 1 is presented first (e.g. as A): ", results_row["present_seg1_first"])
                label_list = cfg.label_list
                print("correct_answer", [label_list[1],label_list[0]][results_row[f"{label_list[0]}_is_{cfg.task_type}"]])
                print("model_answer", results_row["answer"])

            results_filepath = os.path.join(results_folder, f'{suffix}_{cfg.model_name}_results.csv')

            dataframe = pd.DataFrame(all_results).reset_index()
            dataframe.to_csv(results_filepath)
            print(f'This file takes:\n --- {time.time() - time_file_start} seconds ---')

    print(f'--- {time.time() - time_1} seconds ---')
    
    # run analyses (incl. prompt evaluation if cfg.prompt_eval)
    if cfg.run_analysis:
        results_filepath = [os.path.join(results_folder, f'{suffix}_{cfg.model_name}_results.csv') for suffix in cfg.suffixes_to_include]
        accuracy, percentage_A = plot_accuracy(model_name=cfg.model_name,
                                               task_type=cfg.task_type,
                                               in_context=cfg.in_context,
                                               correct_order=cfg.suffixes_to_include,
                                               label_list=cfg.label_list,
                                               output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        log.info(f"{cfg.prompts} Average accuracy for model {cfg.model_name}: {accuracy}")

        with open(os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "avg_accuracy.txt"), "w") as f:
            f.write(str(accuracy))
        
        if cfg.prompt_eval:
            # add prompt eval results to csv  (create one if it does not already exist)
            prompt_config_name = cfg.prompts.name
            prompt_results_csv_path = cfg.prompt_eval_csv.replace(".csv", cfg.model_name) + ".csv"
            if os.path.isfile(prompt_results_csv_path):
                prompt_df = pd.read_csv(prompt_results_csv_path, index_col=0)
                prompt_df = pd.concat([prompt_df, pd.DataFrame([{"model_name": cfg.model_name, 
                                                                 "task_type": cfg.task_type,
                                                                 "prompt_config_name": prompt_config_name,
                                                                 "accuracy": accuracy,
                                                                 "percentageA": percentage_A,
                                                                 "prompt_config": str(cfg.prompts),
                                                                 "path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}])], ignore_index=True)
            else:
                prompt_df = pd.DataFrame([{"model_name": cfg.model_name, 
                                           "task_type": cfg.task_type,
                                           "prompt_config_name": prompt_config_name,
                                           "accuracy": accuracy,
                                           "percentageA": percentage_A,
                                           "prompt_config": str(cfg.prompts),
                                           "path": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}])
            
            prompt_df.to_csv(prompt_results_csv_path)

    # clear GPU memory (important for prompt selection sweep)
    try:
        destroy_model_parallel()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        ray.shutdown()
        del llm
        del tokenizer
        gc.collect()
    except:
        log.error("Could not clear GPU memory properly.")
    return accuracy

if __name__ == '__main__':
    experiment()

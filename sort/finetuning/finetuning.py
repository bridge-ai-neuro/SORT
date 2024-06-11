import pandas as pd
import os
import json
import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from huggingface_hub import login
import torch
import shutil
import numpy as np
import logging
import datetime
import wandb
from accelerate.utils import set_seed

log = logging.getLogger(__name__)
    
@hydra.main(version_base=None,config_path="finetuning_conf", config_name="config")
def finetuning(cfg):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "offline"  # local logging
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    set_seed(1)
    
    # copy this code to the output folder
    current_script_path = __file__
    script_filename = os.path.basename(current_script_path)
    destination= os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, script_filename)
    shutil.copyfile(current_script_path, destination)

    results_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,cfg.result_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    print(f"Results will be saved in {results_folder}")
    wandb.init(project="sort", dir=results_folder)
    model_paths = pd.read_csv(cfg.model_paths_csv, index_col=0)
    model_path = model_paths[model_paths["model_name"]==cfg.model_name].iloc[0]["model_path"]

    print(f"Loading model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code, download_dir=cfg.download_path)
    if "mistral" in cfg.model_name.lower():
        chat_template_name = "mistral.jinja"
    elif "gemma" in cfg.model_name.lower():
        chat_template_name = "gemma.jinja"
    elif "llama" in cfg.model_name.lower() and not "3" in cfg.model_name.lower():
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
    else:
        add_whitespace = False
    log.info(f"Adding a whitespace to the answer for this model: {add_whitespace}")
    
    label_list = [cfg.label_list[0], cfg.label_list[1]] 
    data_dir = cfg.data_path
    files = os.listdir(data_dir)
    files = [f.replace("segments_","").replace("books_","").replace("excerpts_","") for f in files]
    suffixes = list(set(files))
    
    label_list = [cfg.label_list[0], cfg.label_list[1]] #["A", "B"]
    print(f'found data file suffixes: {suffixes}')
    all_prompts = []
    if not cfg.training.use_other_data_instead:
        for suffix in suffixes:
            if suffix in cfg.suffixes_to_include:
                
                books_df = pd.read_csv(os.path.join(data_dir, "books_"+suffix), index_col=0)
                excerpts_df = pd.read_csv(os.path.join(data_dir, "excerpts_"+suffix), index_col=0)
                segments_df = pd.read_csv(os.path.join(data_dir, "segments_"+suffix), index_col=0)
                # filter for only particular books
                segments_df = pd.concat([segments_df[(segments_df["book_idx"]==include_idx)] for include_idx in cfg.books_to_include], axis=0)


                for idx, row in segments_df.iterrows():
                    # only look at the first N excerpts in the data (e.g. to use the last 10 for something else)
                    if row["excerpt_idx"]<cfg.max_excerpt_index and row["excerpt_idx"]>=cfg.min_excerpt_index:
                        book_idx = row["book_idx"]
                        excerpt_idx = row["excerpt_idx"]
                        segment_1 = row["segment_1"]
                        segment_2 = row["segment_2"]
                        A_is_1 = row["present_seg1_first"]
                        excerpt = excerpts_df[(excerpts_df["book_idx"] == book_idx)&(excerpts_df["excerpt_idx"]==excerpt_idx)].iloc[0]["excerpt_text"]
                        book_title = books_df[books_df["book_idx"]==book_idx]["book_title"].iloc[0]

                        if A_is_1:
                            segments_prompt = f"Segment {label_list[0]}: {segment_1} Segment {label_list[1]}: {segment_2}\n"
                            segments = [segment_1, segment_2]
                        else:
                            segments_prompt = f"Segment {label_list[0]}: {segment_2} Segment {label_list[1]}: {segment_1}\n"
                            segments = [segment_2, segment_1]

                        # formulate the full prompt for chat/instruct models with the appropriate template.
                        pre_excerpt_string = cfg.prompts.pre_excerpt.replace("<booktitle>", book_title).replace("<tasktype>", cfg.task_type).replace("<excerpt>",excerpt)
                        pre_excerpt_string = pre_excerpt_string if cfg.in_context else ""
                        post_excerpt_string = cfg.prompts.post_excerpt.replace("<booktitle>", book_title).replace("<tasktype>", cfg.task_type).replace("<segments>", segments_prompt)

                        system_prompt = cfg.prompts.system_prompt.replace("<tasktype>", cfg.task_type)
                        
                        if not cfg.in_context:
                            assert not len(pre_excerpt_string), "excerpt prefix should not be given in no-context condition."

                        # finetune on complete examples
                        if cfg.training.task_finetune:
                            user_prompt = pre_excerpt_string+post_excerpt_string
                            if cfg.use_system_prompt:
                                messages = [{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}]
                            else:
                                messages = [{"role": "user", "content": user_prompt}]

                            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            full_prompt+=cfg.prompts.model_answer_prefix 
                            if add_whitespace: # and the correct answer!
                                full_prompt+=" " + [label_list[1],label_list[0]][A_is_1]
                            else:
                                full_prompt = full_prompt + [label_list[1],label_list[0]][A_is_1]
                                
                        # normal Long term memory condition:
                        else:
                            user_prompt = pre_excerpt_string
                            if cfg.use_system_prompt:
                                messages = [{"role": "system", "content":system_prompt},
                                            {"role": "user", "content": user_prompt}]
                            else:
                                messages = [{"role": "user", "content": user_prompt}]
                            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        # fill in segment labels in prompt
                        full_prompt = full_prompt.replace("<label_list[0]>",label_list[0]).replace("<label_list[1]>", label_list[1])
                        # option to train base model on excerpts
                        if cfg.training.text_only_finetune:
                            full_prompt = pre_excerpt_string

                        # only finetune on prompts that do not exceed the max model size
                        if len(tokenizer.encode(full_prompt)) < cfg.training.max_model_length:
                            all_prompts.append(full_prompt)

    if cfg.training.use_other_data_instead:
        all_prompts = []
        df = pd.read_csv(cfg.training.other_data, index_col=0)
        for value in df.values.tolist():
            all_prompts.append(value[0])

    log.info(f"fine-tuning data: {all_prompts[:10]}")

    # if we mix with additional instruction data:
    if cfg.training.include_instruction_data:
        if cfg.training.only_use_instruction_data:
            max_instruction_samples= len(all_prompts) if not cfg.training.n_instruction_samples else cfg.training.n_instruction_samples
            all_prompts = []
        else:
            max_instruction_samples = len(all_prompts) if not cfg.training.n_instruction_samples else cfg.training.n_instruction_samples
        with open(cfg.training.instruction_data_path, "r") as f:
            instruction_data = json.load(f)
        instruction_data = instruction_data[:max_instruction_samples] # reduce amount to match number of excerpts
        for message_history in instruction_data:
            prompt = tokenizer.apply_chat_template(message_history, tokenize=False, add_generation_prompt=False)
            if len(tokenizer.encode(prompt))<cfg.training.max_model_length*cfg.training.instruction_sample_length_factor:
                all_prompts.append(tokenizer.apply_chat_template(message_history, tokenize=False, add_generation_prompt=False))
            else:
                print(len(tokenizer.encode(prompt)))

    # option to add examples of this task (with excerpts from other books)
    if cfg.training.include_task_examples:
        with open(cfg.training.task_example_path, "r") as f:
            instruction_data = json.load(f)
        for message_history in instruction_data:
            all_prompts.append(tokenizer.apply_chat_template(message_history, tokenize=False, add_generation_prompt=False))

    save_to_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"{cfg.model_name}_{cfg.in_context}.csv")
    tokenizer.pad_token = tokenizer.eos_token
    text_list = pd.DataFrame(all_prompts, columns=["text"])
    text_list.to_csv(save_to_path)
    text_list.index.name = 'idx'
    ds = Dataset.from_dict({'input_ids': text_list["text"]})
    ds = ds.map(lambda batch: {'input_ids': tokenizer(batch['input_ids']).input_ids}, batched=True)

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cfg.download_path,resume_download=True, torch_dtype=torch.bfloat16, use_cache=True, attn_implementation="flash_attention_2") #,device_map = 'auto')
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    training_args = TrainingArguments(
            output_dir=os.path.join(cfg.training.local_output_dir, timestamp, f"finetune_{cfg.model_name}"),#cfg.model_name),
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            fp16=False,
            bf16=cfg.training.bf16,
            learning_rate=cfg.training.lr,
            num_train_epochs=cfg.training.epochs,
            deepspeed=cfg.training.deepspeed,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
            logging_dir=os.path.join(cfg.training.local_output_dir,timestamp,"runs"),
            logging_strategy=cfg.training.logging_strategy,
            eval_steps=0,
            save_strategy=cfg.training.save_strategy,
            save_total_limit=None if not cfg.training.save_total_limit else cfg.training.save_total_limit,
            load_best_model_at_end=False,
            report_to="wandb",
            run_name="finetuning",
            disable_tqdm=False,
            remove_unused_columns=False,
            local_rank=cfg.training.local_rank,
            warmup_steps=cfg.training.warmup_steps,
            adam_beta1=0.9,
            adam_beta2=0.95,
            # min_lr_ratio=1. / 30,
            lr_scheduler_type=cfg.training.lr_schedule,
            adam_epsilon=1e-5,
            weight_decay=cfg.training.weight_decay,
            ddp_backend="nccl",
            max_grad_norm=1.0,
            torch_compile=True
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    log.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator
    )

    log.info("Training")
    trainer.train(resume_from_checkpoint=cfg.training.resume_from_checkpoint)
    savepath = os.path.join(cfg.training.local_output_dir, timestamp, f"finetune_{cfg.model_name}")
    log.info(f"Saving Model to {savepath}")
    trainer.save_model(output_dir=savepath)
    logs_json = json.dumps(trainer.state.log_history)
    with open(os.path.join(cfg.training.local_output_dir, timestamp, "train_logs.json"), "w") as f:
        f.write(logs_json)

    log.info("Done.")

if __name__ == '__main__':
    finetuning()
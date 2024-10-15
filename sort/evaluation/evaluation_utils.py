import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams
from openai import OpenAI
import os
import pandas as pd
import tiktoken
import torch


def load_model_tokenizer(model_path, cfg):
    print(f"Loading model: {model_path}")
    if cfg.api == 'openai':
        client = OpenAI(api_key=cfg.api_key)  # Initialize OpenAI
        llm = [client, model_path]
        tokenizer = tiktoken.encoding_for_model(model_path)
        sampling_params = {'max_tokens': cfg.sample_n_tokens, 'top_logprobs': cfg.n_logprobs,
                           'temperature': 0, 'logprobs': True}
    elif cfg.api == 'vllm':
        # get the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=cfg.trust_remote_code,
                                                  cache_dir=cfg.download_path)
        # get the vocab size
        vocab_size = tokenizer.vocab_size
        if cfg.log_prompt_logprobs:
            sampling_params = SamplingParams(max_tokens=cfg.sample_n_tokens,
                                             # for the next token, get the log probability for the top 10 most probable next tokens
                                             logprobs=cfg.n_logprobs,
                                             # to log the logprob of the prompt (token-wise, including the rank 1 token if it does not align)
                                             prompt_logprobs=1,
                                             # greedy search for the generation
                                             temperature=0)
        else:
            sampling_params = SamplingParams(max_tokens=cfg.sample_n_tokens,
                                             # for the next token, get the log probability for the top 10 most probable next tokens
                                             logprobs=cfg.n_logprobs,
                                             # greedy search for the generation
                                             temperature=0)
        print(f'model_name: {cfg.model_name}')
        # print model loading configs
        print(f'--- Model loading configs ---')
        print(f'model_path: {model_path}')
        print(f'trust_remote_code: {cfg.trust_remote_code}')
        print(f'gpu_memory_utilization: {cfg.gpu_memory_utilization}')
        print(f'tensor_parallel_size: {cfg.tensor_parallel_size}')
        print(f'--- Model loading configs ---')
        llm = LLM(
            model_path,
            trust_remote_code=cfg.trust_remote_code,
            download_dir=cfg.download_path,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_logprobs=32000000,
            #dtype='half',  # DEBUG WITH OLDER HW
            #max_model_len=1e8  # DEBUG WITH GPT-2
        )
        # for debugging, see if a model generates sensible output given a chat template
        if cfg.debugging:
            prompt = tokenizer.apply_chat_template([{"role": "user",
                                                     "content": "What is the distance from the earth to the moon?"}],
                                                   tokenize=False, add_generation_prompt=True)
            output = llm.generate(prompt, sampling_params=sampling_params)
            print(output[0].outputs[0].text)
            exit()
    elif cfg.api == 'hf':
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg.trust_remote_code,
                                                  cache_dir=cfg.download_path)
        llm = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cfg.download_path,
                                                   resume_download=True,
                                                   torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2",
                                                   device_map='auto',
                                                   trust_remote_code=cfg.trust_remote_code)
        sampling_params = GenerationConfig(max_new_tokens=cfg.sample_n_tokens,
                                           output_logits=False,
                                           eos_token_id=tokenizer.eos_token_id,
                                           pad_token_id=tokenizer.pad_token_id)
    return tokenizer, sampling_params, llm


def llm_generate(llm, batch_prompts, sampling_params, api, tokenizer):
    if api == 'openai':
        client, model_name = llm
        assert len(batch_prompts) == 1, "Batch size expected to be 1!"
        max_retries = 5
        i = 0
        n_wait = 2
        while i < max_retries:
            try:
                generate_outputs = [client.chat.completions.create(
                    model=model_name, messages=batch_prompts[0], **sampling_params)]
                if generate_outputs[0]:
                    break
            except Exception as e:
                i += 1
                # Exponential backoff. Gets longer and longer each time.
                n_wait *= 2 
                print(f"Encountered error `{e}`. Trying again in {n_wait}s...")
                time.sleep(n_wait)
                continue
        if i >= max_retries:
            raise ValueError("Exiting, unable to get chat completion response!")
    elif api == 'vllm':
        generate_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
    elif api == 'hf':
        generate_outputs = huggingface_inference(llm, tokenizer, batch_prompts, sampling_params)
    return generate_outputs


def parse_output(generation_output, tokenizer, api, response_tokens):
    # Returns:
    #   model_response: the new generated text
    #   prob_list: log probabilities associated with the 2 correct response tokens
    if api == 'openai':
        assert len(generation_output.choices) == 1, "Too many responses!"
        model_response = generation_output.choices[0].message.content
        # probabilities below are given as a list of the top P token probabilities,
        # as TopLogProb objects that contain: token, bytes, logprob
        # generation_output.choices[0].logprobs.content contains logprobs for all tokens
        # when we get the first one, we get the top N logprobs for the first token
        tok0_logprobs = generation_output.choices[0].logprobs.content[0].top_logprobs
        tok0_candidates = [t.token for t in tok0_logprobs]
        probabilities = get_answer_prob(tok0_logprobs, tok0_candidates, response_tokens, api='openai')
        # probabilities = generation_output.choices[0].logprobs.content[0].top_logprobs
        # Should just get probabilities associated with the responses
    elif api == 'vllm':
        model_response = generation_output.outputs[0].text
        # generation_output contains:
        # prompt_logprobs: len n_prompt_token list containing logprobs associated with each prompt tokens.
        #                  will include additional if it does not match the true token (sampling_params)
        # outputs[0].logprobs: len new_prompt_token list containing logprobs for top cfg.n_logprobs at each token
        #                      position. we only look at the first token
        # NOTE: the log probabilities using the vLLM API differ from the HF API, although answer parsing typically
        # gives the same output. Unclear why the log probabilities should differ for the same model.
        tok0_logprobs = [v for v in generation_output.outputs[0].logprobs[0].values()]
        tok0_candidates = [t.decoded_token for t in tok0_logprobs]
        probabilities = get_answer_prob(tok0_logprobs, tok0_candidates, response_tokens, api='vllm')
    elif api == 'hf':
        model_response = generation_output['raw_answer'][0]  # assuming batch size 1
        tmp = [t for rt in response_tokens for t in rt]
        tok0_candidates = {t: tokenizer.encode(t)[-1] for t in tmp}
        tok0_logprobs = generation_output['answer_logprobs']  # logprobs for first token (array of vocab size)
        probabilities = get_answer_prob(tok0_logprobs, tok0_candidates, response_tokens, api='hf')
    else:
        raise ValueError(f'api {api} needs to be in openai, vllm, or hf')

    probabilities = [p.logprob if not isinstance(p, float) else p for p in probabilities]
    return model_response, np.array(probabilities)


def parse_for_results(generation_output, tokenizer, cfg, results_row, response_tokens, bos_token=False):
    # Answer based on log probabilities
    model_response, prob_list = parse_output(generation_output, tokenizer, cfg.api, response_tokens)
    results_row["prob_A"] = prob_list[0]
    results_row["prob_B"] = prob_list[1]
    # Answer based on text parsing
    results_row = results_row.drop(["segment_1", "segment_2"])
    results_row["raw_answer"] = model_response
    results_row["model_name"] = cfg.model_name
    results_row[f"{cfg.label_list[0]}_is_{cfg.task_type}"] = results_row[
        "present_seg1_first"] if "first" in cfg.task_type else int(not results_row[
        "present_seg1_first"])  # gives an index such that ["B", "A"][idx] is the correct answer for the task

    # parse the answer from probabilities.
    if (prob_list[0] == prob_list[1]) and (prob_list[0] < -0.5e12):
        # if the answer didn't contain any appropriate response tokens
        prob_answer = 'X'
    else:
        # otherwise (i.e. in most cases), set the higher probability as the model answer
        prob_answer = cfg.label_list[int(np.argmax(prob_list))]
    results_row["answer"] = prob_answer

    if (cfg.api == 'openai') or (not cfg.log_prompt_logprobs):
        # OpenAI ChatCompletions API does not support `echo` argument which returns the prompt logprobs.
        results_row["prompt_logprob_jsons"] = '{}'
        if cfg.api == 'openai':
            results_row["token_count"] = generation_output.usage.prompt_tokens
            model_answer = parse_str_response_openai(model_response, cfg.label_list)
            results_row["parsed_raw_answer"] = model_answer
    elif cfg.api == 'hf' and cfg.log_prompt_logprobs:
        model_answer = parse_str_response(model_response, cfg.label_list)
        results_row["parsed_raw_answer"] = model_answer
        results_row["prompt_logprob_jsons"] = generation_output['json_logprobs']
        results_row["token_count"] = generation_output['token_count']
    elif cfg.api == 'vllm' and cfg.log_prompt_logprobs:
        model_answer = parse_str_response(model_response, cfg.label_list)
        results_row["parsed_raw_answer"] = model_answer
        results_row["prompt_logprob_jsons"] = logprob_logging(generation_output, bos_token=bos_token)
        results_row["token_count"] = len(generation_output.prompt_token_ids)
    return results_row


def huggingface_inference(model, tokenizer, text, gcfg):
    """
    Takes a transformers/huggingface model, its tokenizer and a string and evaluates the probability of
    """
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    token_count = input_ids.input_ids.shape[-1]
    # Old version was only generating a single token. Let's actually generate multiple tokens with the built-in
    # generate() functionality.
    truncate_length = tokenizer.model_max_length - gcfg.max_new_tokens
    truncated = False
    if token_count > truncate_length:
        truncated = True
        # POTENTIALLY TRUNCATES THE QUESTION
        print(f"Exceeding maximum context length, truncating input...")
        input_ids['input_ids'] = input_ids['input_ids'][:, :truncate_length]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, :truncate_length]
    pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
 
    if gcfg.max_new_tokens > 1:
       out = model.generate(inputs=input_ids.input_ids, attention_mask=input_ids.attention_mask,
                             generation_config=gcfg, return_dict_in_generate=True, pad_token_id=pad_token)
        # Now decode the text of the generated tokens
       gen_tokens = out.sequences[:, token_count:]
       predicted_answer = tokenizer.batch_decode(gen_tokens)

    with torch.no_grad():
        outputs = model(**input_ids)
    logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    # Get the ids of the predicted token across the entire sequence (starting with prediction based on the first token)
    rank_1_token_ids = logprobs.argmax(-1)  # get the index of the most probable token across entire sequence
    # Get answer from model incl. logprobs for the answer across entire vocab
    if gcfg.max_new_tokens == 1:
        pred_tokens = rank_1_token_ids[:, -1]  # Not needed -- only place this is used is when decoding text
        predicted_answer = tokenizer.batch_decode(pred_tokens)
    answer_logprobs = logprobs[0, -1, :]  # Not needed -- covered above
    # Get the logprobs for LM on the prompt tokens -- these may differ from the true inputs!
    prompt_log_probs = logprobs[
        0, np.arange(0, (token_count - 1) if not truncated else truncate_length - 1),
        input_ids.input_ids[:, 1:]]  # get the log probs associated with the true token
    rank_1_decoded = [tokenizer.decode(x) for x in rank_1_token_ids[0, :-1]]
    accuracy = (rank_1_token_ids[:, :-1] == input_ids.input_ids[:, 1:])
    # move results to cpu as numpy arrays. also casting to float in case you're using bf16
    prompt_log_probs = prompt_log_probs.cpu().float().numpy()[0]
    accuracy = accuracy.cpu().float().numpy()[0]
    answer_logprobs = answer_logprobs.cpu().numpy()
    input_ids = input_ids.input_ids.cpu().numpy()[0, 1:]  # first token does not have any logprobs
    # log the logprob of the prompt, whether it was rank 1, what it decodes to and what the rank 1 prediction decodes to
    logprobdicts = [{int(input_ids[idx]): {"logprob": float(prompt_log_probs[idx]),
                                           "rank_1": int(accuracy[idx]),
                                           "decoded_token": tokenizer.decode(input_ids[idx]),
                                           "decoded_token_rank1": rank_1_decoded[idx]}} for idx in
                    range(0, len(accuracy))]

    # encode list of dicts as string
    json_logprobs = json.dumps(logprobdicts)

    return {"raw_answer": predicted_answer, "answer_logprobs": answer_logprobs,
            "json_logprobs": json_logprobs, "token_count": token_count}


def logprob_logging(output, bos_token=True):
    """
    Specific to vLLM (>=0.4.0), to log the log probability of all prompt tokens.
    Requires the argument 'prompt_logprobs' to be set to True during generation and logprobs to be equal to 1.

    Inputs:
        output (single vLLM RequestOutput)
    """
    start_token = int(bos_token)  # skip first token if it is <bos>
    if output.prompt_logprobs[start_token] is None:
        start_token += 1

    prompt_tokens = output.prompt_token_ids
    accuracy = [1 if (logprob_dict[token_id].rank==1) else 0 \
                for logprob_dict, token_id in zip(output.prompt_logprobs[start_token:],
                                                  prompt_tokens[start_token:])
                ]

    logprobs = []
    for logprob_dict, token_id in zip(output.prompt_logprobs[start_token:],
                                      prompt_tokens[start_token:]):
        toks = list(logprob_dict.keys())
        rank_1_tok = toks[0] if logprob_dict[toks[0]].rank==1 else toks[1]
        outd = {token_id: {"logprob": logprob_dict[token_id].logprob,
                            "rank_1": 1 if rank_1_tok==token_id else 0,
                            "decoded_token": logprob_dict[token_id].decoded_token,
                            "decoded_token_rank1": logprob_dict[rank_1_tok].decoded_token}}
        logprobs.append(outd)

    json_logprobs = json.dumps(logprobs)
    return json_logprobs


def parse_str_response(response_text, label_list):
    """
    Check if answer token contains one of the labels and not the other to get the answer.
    response_text: str
    label_list: list of the labels (e.g. ["A","B"])
    """
    if label_list[0] in response_text and not label_list[1] in response_text:
        answer = label_list[0]
    elif label_list[1] in response_text and not label_list[0] in response_text:
        answer = label_list[1]
    else:
        answer = "X"
        print(f"Answer {response_text} could not be matched to either {label_list[0]} or {label_list[1]}")
    return answer


def parse_str_response_openai(resp_text, label_list):
    """
    Check if answer token contains one of the labels and not the other to get the answer.
    response_text: str
    label_list: list of the labels (e.g. ["A","B"])
    """
    resp_text = resp_text.strip()
    lower = [r.lower() for r in label_list]
    if len(resp_text) == 1 and resp_text in label_list or resp_text in lower:
        this_ans = resp_text
    elif label_list[0] in resp_text and not label_list[1] in resp_text:
        this_ans = label_list[0]
    elif label_list[1] in resp_text and not label_list[0] in resp_text:
        this_ans = label_list[1]
    elif any([lab + ":" in resp_text for lab in label_list]):
        for i, lab in enumerate(label_list):
            if lab + ":" in resp_text:
                this_ans = label_list[i]
    else:
        this_ans = 'X'
        print(f'Could not parse answer in response text: {resp_text}') 
 
    return this_ans


def get_answer_prob(logprobs, tokens, answer_list, api):
    """ Get the log probability for tokens in the answer list.
    Inputs:
        logprobs: log probabilities for generated tokens. For vLLM or OpenAI, these are the custom
            LogProb class objects for that API. For HuggingFace, this should just be a numpy array
            of the log probabilities associated with the
        tokens: the generated tokens. should be the same length as logprobs. For vLLM or OpenAI, this is
            simply a list of strings. For HuggingFace this is a dictionary where the token keys are the
            strings, and the values are the tokenized IDs.
        answer_list: a 2 element list containing [answer_1_tokens, answer_2_tokens]. each element
                     of this list can be a list to accommodate multiple possible output tokens that
                     should be counted as correct (e.g., 'A' and ' A').
    Outputs:
        prob_list: a 2 element list containing the log probabilities (either objects or numbers)
                   that are associated with the possible answers. Can compute accuracy by finding
                   the highest logprob.
    """
    if api == 'hf':
        tok_str, tok_id = list(tokens.keys()), list(tokens.values())
    else:
        tok_str = tokens
    prob_list = []
    for answer in answer_list:
        # To account for multiple possible answer tokens, e.g. 'A' or ' A', iterate over the
        # possible answers. Assumes that the first option in answer_list is the preferred option
        matching_token_inds = [tok_str.index(a) for a in answer if a in tok_str]
        if len(matching_token_inds) == 0:
            prob_list.append(LogProb(value=-1e12))
        else:
            if api == 'hf':
                # logprobs is just a large numpy array of the whole vocabulary, so find the token
                # index corresponding to the desired answer token
                tok_ind = tok_id[matching_token_inds[0]]
                prob_list.append(float(logprobs[tok_ind]))  # force np float to python float
            else:
                prob_list.append(logprobs[matching_token_inds[0]])

    return prob_list


class LogProb():
    def __init__(self, value):
        self.logprob = value


def calc_accuracy(task_type, label_list, output_dir):
    """
    task_type: "first" or "second"
    label_list: e.g. ["A", "B"]
    """
    folder_path = f"{output_dir}/results"
    files = os.listdir(folder_path)
    corrects = []
    A_frequencies = []

    for file in files:
        results_name = os.path.join(folder_path,file)
        df = pd.read_csv(results_name, index_col=0)
        df["ground_truth"] = df[f"{label_list[0]}_is_{task_type}"].apply(lambda x: [label_list[1],label_list[0]][x])
        df["correct"] = df["ground_truth"] == df["answer"]
        corrects.append((file.split(".csv")[0], df["correct"].values))
        A_frequencies.append(df["answer"].describe()["freq"]/len(df))
    means = []
    for data_list in corrects:
        mean = np.mean(data_list[1])
        means.append(mean)

    return np.mean(means), np.mean(A_frequencies)

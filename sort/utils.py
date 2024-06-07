import torch

def get_models_plot_info():
    it_str = '-inst'  # instruction tuned suffix for model names.

    models_plot_info = {
        "mistral-instruct-7b": {
            "name": "mistral-7b-v1",
            "label": "Mistral-v1-7b" + it_str,
            "color": "cornflowerblue"
        },
        "mistral-instruct-7b-v2": {
            "name": "mistral-7b-instruct-v2",
            "label": "Mistral-v2-7b" + it_str,
            "color": "teal"
        },
        "llama3-8b-instruct":  {
            "name": "llama3-8b",
            "label":  "Llama3-8b"+ it_str,
            "color": "darkorange"
        },
        "gemma7b_1.1_inst": {
            "name": "gemma-7b-1.1",
            "label": "Gemma-1.1-7b" + it_str,
            "color": "red"
        },
        "Nous-Hermes-2-Mixtral-8x7B-DPO": {
            "name": "mistral_8x7b_dpo",
            "label": "Mixtral-8x7b-DPO" + it_str,
            "color": "darkgoldenrod"
        },
        "Mixtral-8x22b": {
            "name": "mixtral-8x22",
            "label": "Mixtral-8x22b" + it_str,
            "color": "deepskyblue"},  # fixed
        "llama2_7b-instruct": {
            "name": "llama2-7b",
            "label": "Llama2-7b" + it_str,
            "color": "darkviolet"
        },
        "llama2_70b-instruct": {
            "name": "llama2-70b",
            "label": "Llama2-70b" + it_str,
            "color": "magenta"
        },
        "llama3_70b-instruct": {
            "name": "llama3-70b",
            "label": "Llama3-70b" + it_str,
            "color": "hotpink"
        },
        "gpt3-5": {
            "name": "gpt-3.5",
            "label": "GPT-3.5-turbo",
            "color": "brown"
        },
        "gpt4": {
            "name": "gpt-4",
            "label": "GPT-4",
            "color": "olive"
        },
    }
    return models_plot_info

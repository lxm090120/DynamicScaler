def select_prompt_from_multi_prompt_dict_by_factor(prompt_dict, factor):
    assert 0.0 <= factor <= 1.0, f"select_prompt: input factor {factor} not legal"
    sorted_keys = list(sorted(prompt_dict.keys()))
    for key in sorted_keys:
        if factor <= key:
            return prompt_dict[key]
    return prompt_dict[sorted_keys[-1]]  # 如果factor大于所有key，返回最后一个prompt

if __name__ == "__main__":
    prompt_dict = {
        0.5: 0.5,
        1.0: 1.0,
    }
    for i in range(10):
        print(i/10, select_prompt_from_multi_prompt_dict_by_factor(prompt_dict, i/10))
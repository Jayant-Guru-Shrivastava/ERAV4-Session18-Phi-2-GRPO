def length_reward(prompts, completions, **kwargs):
    return [len(c) / 100.0 for c in completions]

def merge_conversation(conversation):
    valid_pairs = [(q, a) for q, a in conversation if a is not None]
    formatted_pairs = [f"{q} {a}." for q, a in valid_pairs]
    result = ["-".join(formatted_pairs) + "\n"]
    return result
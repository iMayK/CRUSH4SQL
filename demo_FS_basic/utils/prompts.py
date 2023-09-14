BASE = (
    'Please write SQL for "{}" based on the following schema:\n\n {} \n\n'
    'Question: {}\n'
    'SQL:'
    )

FEW_SHOT = (
    'Please write SQL for "{}" based on the following schema:\n\n {} \n\n'
    'Here are some examples of correct text-sql pairs for your reference:\n\n{} \n'
    'Question: {}\n'
    'SQL:'
    )

PROMPTS = {
    "base": BASE,
    "fewshot": FEW_SHOT,
}



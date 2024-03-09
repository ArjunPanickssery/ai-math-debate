from transformers import AutoTokenizer, AutoModelForCausalLM
from data import load_data, DatasetItem
import random

DEBATER_MODEL_NAME = "WizardLM/WizardMath-70B-V1.0"
JUDGE_MODEL_NAME = "WizardLM/WizardMath-70B-V1.0"


def format_wizard_math_prompt(question):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"


def tokenize_input(text, tokenizer):
    return tokenizer.encode(text, return_tensors="pt")


def generate_response(
    prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
):
    input_ids = tokenize_input(prompt, tokenizer)
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def make_proof_prompt(
    question: str, proof_a: str, proof_b: str, justify_letter: str, justify_numeric: str
):
    return f"""Below are two proofs that claim to solve the question: {question}

Explain why proof {justify_letter} for answer {justify_numeric} is correct.

(A) {proof_a}

(B) {proof_b}
"""


def get_debater_questions(dataset_item: DatasetItem):
    """
    Both proofs together make up the question - which one is correct
    """
    a, b = dataset_item.answer_correct.proof, dataset_item.answer_incorrect.proof
    correct_letter, incorrect_letter = "A", "B"
    if random.random() > 0.5:
        # Randomize the order of the proofs
        a, b = b, a
        correct_letter, incorrect_letter = incorrect_letter, correct_letter
    question_correct = make_proof_prompt(
        dataset_item.question, a, b, correct_letter, dataset_item.answer_correct.numeric
    )
    question_incorrect = make_proof_prompt(
        dataset_item.question,
        a,
        b,
        incorrect_letter,
        dataset_item.answer_incorrect.numeric,
    )
    return question_correct, question_incorrect


def main():
    tokenizer = AutoTokenizer.from_pretrained(DEBATER_MODEL_NAME)
    train_data, test_data = load_data()
    debater_a = debater_b = AutoModelForCausalLM.from_pretrained(
        DEBATER_MODEL_NAME, load_in_8bit=True
    )
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME, load_in_8bit=True
    )
    for item in train_data:
        prompt_correct, prompt_incorrect = get_debater_questions(item)
        response_correct = generate_response(prompt_correct, debater_a, tokenizer)
        response_incorrect = generate_response(prompt_incorrect, debater_b, tokenizer)

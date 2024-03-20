import os

os.environ["HF_HOME"] = "/scratch/bbry/apanickssery/"

from prompts import BLIND_JUDGE_PROMPT, JUDGE_PROMPT, MAKE_ARGUMENT_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data import load_data, DatasetItem, save_to_json
from typing import Callable
from dotenv import load_dotenv
from dotenv import load_dotenv
import random
import torch

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

WIZARDMATH_NAME = "WizardLM/WizardMath-70B-V1.0"
LLAMA7B_NAME = "meta-llama/Llama-2-7b-chat-hf"
LLAMA70B_NAME = "meta-llama/Llama-2-70b-chat-hf"
MATH_MISTRAL_NAME = "meta-math/MetaMath-Mistral-7B"
ARITHMO_NAME = "upaya07/Arithmo2-Mistral-7B"  # base model
# LLEMMA_NAME = "EleutherAI/llemma_34b" # base model

DEBATER_MODEL_NAME = WIZARDMATH_NAME
JUDGE_MODEL_NAME = MATH_MISTRAL_NAME


def format_wizard_math_prompt(question):
    """
    This comes from Huggingface
    https://huggingface.co/WizardLM/WizardMath-70B-V1.0
    """
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"


def format_llama_chat_prompt(question):
    return f"""<s>[INST] <<SYS>>
You are a judge of debates. You use your best judgment to make final decisions between the arguments of debaters.
<</SYS>>

{question} [/INST]"""


def format_math_mistral_prompt(question):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:"""


# Custom prompt format functions for different models
# E.g. instruction formatting
FORMAT_FUNCTIONS = {
    WIZARDMATH_NAME: format_wizard_math_prompt,
    LLAMA7B_NAME: format_llama_chat_prompt,
    LLAMA70B_NAME: format_llama_chat_prompt,
    MATH_MISTRAL_NAME: format_math_mistral_prompt,
}
MODEL_TO_STRING = {
    WIZARDMATH_NAME: "wizardmath70b",
    LLAMA7B_NAME: "llama7b",
    LLAMA70B_NAME: "llama70b",
    MATH_MISTRAL_NAME: "math_mistral7b",
}


def generate_response(
    question: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_format_fn: Callable[[str], str] = lambda x: x,
):
    full_prompt = prompt_format_fn(question)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    output = model.generate(input_ids.to("cuda"), max_new_tokens=500, temperature=0)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # response = decoded.split("Response:")[1].strip()
    response = decoded[len(full_prompt) :].strip()
    return response


def get_probs(
    question: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_format_fn: Callable[[str], str] = lambda x: x,
):
    full_prompt = prompt_format_fn(question)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    output = model(input_ids).logits[0, -1, :]
    probs = output.softmax(dim=0)
    return probs


def get_debater_questions(dataset_item: DatasetItem):
    """
    Both proofs together make up the question
    """
    proof_a, proof_b = (
        dataset_item["answer_correct"]["proof"],
        dataset_item["answer_incorrect"]["proof"],
    )
    correct_letter, incorrect_letter = "A", "B"

    # Randomize the order of the proofs so that the judge doesn't learn to always pick the first one
    if random.random() > 0.5:
        proof_a, proof_b = proof_b, proof_a
        correct_letter, incorrect_letter = incorrect_letter, correct_letter

    question_correct = MAKE_ARGUMENT_PROMPT.format(
        question=dataset_item["question"],
        justify_letter=correct_letter,
        justify_numeric=dataset_item["answer_correct"]["numeric"],
        proof_a=proof_a,
        proof_b=proof_b,
    )
    question_incorrect = MAKE_ARGUMENT_PROMPT.format(
        question=dataset_item["question"],
        justify_letter=incorrect_letter,
        justify_numeric=dataset_item["answer_incorrect"]["numeric"],
        proof_a=proof_a,
        proof_b=proof_b,
    )

    return question_correct, question_incorrect, (correct_letter == "A")


def get_judge_question(
    dataset_item: DatasetItem,
    correct_debater_response: str,
    incorrect_debater_response: str,
    is_a_correct: bool,
    is_blind: bool = False,
):
    # Whether A or B is correct was randomized in get_debater_questions
    # So we need `is_a_correct` to determine which of the correct or incorrect answers / responses to use
    if is_a_correct:
        numeric_a, proof_a, argument_a = (
            dataset_item["answer_correct"]["numeric"],
            dataset_item["answer_correct"]["proof"],
            correct_debater_response,
        )
        numeric_b, proof_b, argument_b = (
            dataset_item["answer_incorrect"]["numeric"],
            dataset_item["answer_incorrect"]["proof"],
            incorrect_debater_response,
        )
    else:
        numeric_a, proof_a, argument_a = (
            dataset_item["answer_incorrect"]["numeric"],
            dataset_item["answer_incorrect"]["proof"],
            incorrect_debater_response,
        )
        numeric_b, proof_b, argument_b = (
            dataset_item["answer_correct"]["numeric"],
            dataset_item["answer_correct"]["proof"],
            correct_debater_response,
        )

    if is_blind:  # blind judge doesn't see arguments
        return BLIND_JUDGE_PROMPT.format(
            question=dataset_item["question"],
            numeric_a=numeric_a,
            proof_a=proof_a,
            numeric_b=numeric_b,
            proof_b=proof_b,
        )

    return JUDGE_PROMPT.format(
        question=dataset_item["question"],
        numeric_a=numeric_a,
        proof_a=proof_a,
        numeric_b=numeric_b,
        proof_b=proof_b,
        argument_a=argument_a,
        argument_b=argument_b,
    )


def main():
    outputs = []
    print(f"Device count: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(DEBATER_MODEL_NAME)
    train_data, test_data = load_data()

    print("Loading debater ...")
    debater_one = debater_two = AutoModelForCausalLM.from_pretrained(
        DEBATER_MODEL_NAME,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    print("Loading judge ...")
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    print("Loaded models.")

    a_token = tokenizer.encode("A")[-1]
    b_token = tokenizer.encode("B")[-1]
    for item in train_data[:100]:
        question_correct, question_incorrect, is_proof_a_correct = (
            get_debater_questions(item)
        )
        # Response of debater tasked with justifying the correct answer
        response_correct = generate_response(
            question_correct,
            debater_one,
            tokenizer,
            prompt_format_fn=FORMAT_FUNCTIONS[DEBATER_MODEL_NAME],
        )
        # Response of debater tasked with justifying the incorrect answer
        response_incorrect = generate_response(
            question_incorrect,
            debater_two,
            tokenizer,
            prompt_format_fn=FORMAT_FUNCTIONS[DEBATER_MODEL_NAME],
        )
        judge_question = get_judge_question(
            item, response_correct, response_incorrect, is_proof_a_correct
        )
        blind_judge_question = get_judge_question(
            item,
            response_correct,
            response_incorrect,
            is_proof_a_correct,
            # Blind judge does not get to see the justifications
            is_blind=True,
        )
        judge_probs = get_probs(
            judge_question,
            judge_model,
            tokenizer,
            # To prime it to predict tokens A or B
            prompt_format_fn=lambda x: FORMAT_FUNCTIONS[JUDGE_MODEL_NAME](x) + "\n(",
        )
        correct_judge_prob = (
            judge_probs[a_token] if is_proof_a_correct else judge_probs[b_token]
        )
        incorrect_judge_prob = (
            judge_probs[b_token] if is_proof_a_correct else judge_probs[a_token]
        )

        blind_judge_probs = get_probs(
            blind_judge_question,
            judge_model,
            tokenizer,
            # To prime it to predict tokens A or B
            prompt_format_fn=lambda x: FORMAT_FUNCTIONS[JUDGE_MODEL_NAME](x) + "\n(",
        )
        correct_blind_judge_prob = (
            blind_judge_probs[a_token]
            if is_proof_a_correct
            else blind_judge_probs[b_token]
        )
        incorrect_blind_judge_prob = (
            blind_judge_probs[b_token]
            if is_proof_a_correct
            else blind_judge_probs[a_token]
        )

        output = f"""
[ORIGINAL DATA]
Question: {item['question']}
Correct proof: {item['answer_correct']['proof']}
Incorrect proof: {item['answer_incorrect']['proof']}

[RESPONSE OF DEBATER JUSTIFYING CORRECT ANSWER]
{response_correct}

[RESPONSE OF DEBATER JUSTIFYING INCORRECT ANSWER]
{response_incorrect}

[INPUT TO JUDGE]
{judge_question}

[PROBABILITIES OF JUDGE]
Probability given to correct answer {item['answer_correct']['numeric']}: {correct_judge_prob*100:.2f}%
Probability given to incorrect answer {item['answer_incorrect']['numeric']}: {incorrect_judge_prob*100:.2f}%

[INPUT TO BLIND JUDGE]
{blind_judge_question}

[PROBABILITIES OF BLIND JUDGE]
Probability given to correct answer {item['answer_correct']['numeric']}: {correct_blind_judge_prob*100:.2f}%
Probability given to incorrect answer {item['answer_incorrect']['numeric']}: {incorrect_blind_judge_prob*100:.2f}%    
"""
        print(output)
        outputs.append(output)

    save_to_json(outputs, "outputs.json")


if __name__ == "__main__":
    main()

import json
import re
from typing import TypedDict
import os

import anthropic
import dotenv
from datasets import load_dataset

dotenv.load_dotenv()
client = anthropic.Anthropic()

GENERATE_WRONG_ANSWER_TEMPLATE = """Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.

<question>{question}</question>

<proof>
{proof}</proof>"""


class Answer(TypedDict):
    numeric: float
    proof: str


class DatasetItem(TypedDict):
    question: str
    answer_correct: Answer
    answer_incorrect: Answer


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def make_correct_answer(item: dict) -> Answer:
    return {
        "numeric": float(item["answer"].split("#")[-1].replace(",", "").strip()),
        "proof": remove_bracketed_math_from_string(item["answer"]),
    }


def make_incorrect_answer(item: dict) -> Answer:
    return ""
    response = generate_wrong_proof(item["question"], item["answer_correct"]["proof"])
    return {"numeric": float(response.split("#")[-1].strip()), "proof": response}


def reformat_dataset_item(item: dict) -> DatasetItem:
    item["answer_correct"] = make_correct_answer(item)
    item["answer_incorrect"] = make_incorrect_answer(item)
    del item["answer"]
    return item


def load_data():
    return load_from_json("train_data.json"), load_from_json("test_data.json")


def load_huggingface_data() -> tuple[list[DatasetItem], list[DatasetItem]]:
    dataset = load_dataset("gsm8k", "main")
    train_data, test_data = list(dataset["train"]), list(dataset["test"])

    train_data = [reformat_dataset_item(item) for item in train_data]
    test_data = [reformat_dataset_item(item) for item in test_data]

    return train_data, test_data


def remove_bracketed_math_from_string(s):
    """Removes bracketed math expressions from a string, e.g. Mike will get 2*$500 = $<<2*500=1000>>1000."""
    return re.sub(r"<<.*?>>", "", s)


def extract_proof_from_string(s):
    """Returns text between <proof> and </proof> tags"""
    return s.split("<proof>")[1].split("</proof>")[0].strip()


def generate_wrong_proof(question, correct_proof):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0,
        system="You are an expert math educator. Your task is to take math proofs and introduce small errors in the logic that make the final answer wrong. You can introduce an error in reasoning or intermediate steps that is small, as long as the final answer is different. This will be very helpful and even essential for our teaching work in the classroom.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</question>\n\n<proof>\nNatalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72\n</proof>",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nNatalia sold 48/2 = 24 clips in May.\nNatalia sold 48-24 = 24 clips altogether in April and May.\n#### 24\n</proof>",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>It takes Roque two hours to walk to work and one hour to ride his bike to work. Roque walks to and from work three times a week and rides his bike to and from work twice a week. How many hours in total does he take to get to and from work a week with walking and biking?\n</question>\n\n<proof>\nRoque takes 2*3 = 6 hours a week to walk to work.\nRoque takes 6*2 = 12 hours a week to walk to and from work.\nRoque takes 1*2 = 2 hours a week to bike to work.\nRoque takes 2*2 = 4 hours a week to bike to and from work.\nIn total, Roque takes 12+4 = 16 hour a week to go to and from work.\n#### 16\n</proof>",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nRoque takes 2*3 = 6 hours a week to walk to work.\nRoque takes 6*2 = 12 hours a week to walk to and from work.\nRoque takes 1*2 = 2 hours a week to bike to work.\nIn total, Roque takes 12+2 = 14 hour a week to go to and from work.\n#### 14\n</proof>",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?</question>\n\n<proof>\nWeng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10 \n</proof> ",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nWeng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 50/0.2 = $250.\n#### 250\n</proof> ",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. If he can bike at 25 mph how much time does he spend biking a week?</question>\n\n<proof>\nHe bikes 20*2=40 miles each day for work\nSo he bikes 40*5=200 miles for work\nThat means he bikes a total of 200+200=400 miles for work\nSo he bikes a total of 400/25=16 hours\n#### 16\n</proof>",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nHe bikes 20*2=40 miles each day for work\nSo he bikes 40*5=100 miles for work\nThat means he bikes a total of 100+100=200 miles for work\nSo he bikes a total of 200/25=8 hours\n#### 8\n</proof>",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?</question>\n\n<proof>\nIn the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5\n</proof>",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nIn the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $15 more.\n#### 15\n</proof>",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a proof. Introduce a small error in the reasoning that makes the final answer wrong. Enclose the new (errant) reasoning in <proof></proof> tags.\n\n<question>Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?</question>\n\n<proof>\nA pound of cream cheese cost $10 / 2 = $5.\nA pack of cold cuts cost $10 x 2 = $20.\nJasper spent $10 + $5 + $20 = $35 on the ingredients.\n#### 35\n</proof>",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<proof>\nA pound of cream cheese cost $20 / 2 = $10.\nA pack of cold cuts cost $10 x 2 = $20.\nJasper spent $20 + $10 + $20 = $50 on the ingredients.\n#### 35\n</proof>",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": GENERATE_WRONG_ANSWER_TEMPLATE.format(
                            question=question, proof=correct_proof
                        ),
                    }
                ],
            },
        ],
    )
    return extract_proof_from_string(message.content[0].text)

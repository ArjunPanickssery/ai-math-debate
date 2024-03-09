from datasets import load_dataset
from typing import TypedDict


class Answer(TypedDict):
    numeric: float
    proof: str


class DatasetItem(TypedDict):
    question: str
    answer_correct: Answer
    answer_incorrect: Answer


def make_correct_answer(item: dict) -> Answer:
    return {
        "numeric": float(item["answer"].split("#")[-1].replace(",", "").strip()),
        "proof": item["answer"],
    }


def make_incorrect_answer(item: dict) -> Answer:
    raise NotImplementedError("TODO: fill in with Claude 3")


def reformat_dataset_item(item: dict) -> DatasetItem:
    item["answer_correct"] = make_correct_answer(item)
    item["answer_incorrect"] = make_incorrect_answer(item)
    del item["answer"]
    return item


def load_data() -> tuple[list[DatasetItem], list[DatasetItem]]:
    dataset = load_dataset("gsm8k", "main")
    train_data, test_data = list(dataset["train"]), list(dataset["test"])

    train_data = [reformat_dataset_item(item) for item in train_data]
    test_data = [reformat_dataset_item(item) for item in test_data]

    return train_data, test_data

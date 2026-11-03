from dataclasses import dataclass
from typing import Any
from datasets import load_dataset


class ProblemRecord:
    """A single problem with a stable ID, question, and reference answer."""

    def __init__(self, id: str, question: str, answer: str) -> None:
        self.id = id
        self.question = question
        self.answer = answer


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset provenance and the exact records selected for a run."""

    dataset_id: str | None
    dataset_config: str | None
    revision: str | None
    split_names: tuple[str, ...]
    selected_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str] | None]:
        """Return metadata in a format that can be serialized as JSON."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "revision": self.revision,
            "split_names": list(self.split_names),
            "selected_ids": list(self.selected_ids),
        }


# intermediate structure for loading datasets from Hugging Face, before filtering by split
class LoadedDataset:
    """
    A collection of problems loaded from Hugging Face, organized by split (e.g., train, validation, test).
    Each split contains a list of ProblemRecords with stable IDs, questions, and reference answers.
    """

    def __init__(
        self,
        problem_splits: dict[str, list[ProblemRecord]],
        dataset_id: str | None = None,
        dataset_config: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.problem_splits = problem_splits
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.revision = revision

    @classmethod
    def load_from_huggingface(
        cls,
        dataset_id: str,
        dataset_config: str | None = None,
        revision: str | None = None,
    ) -> "LoadedDataset":
        """Load a dataset and retain the requested Hugging Face provenance."""
        raw_dataset = load_dataset(dataset_id, name=dataset_config, revision=revision)
        problem_splits = {}
        for split in raw_dataset.keys():
            problem_splits[split] = []
            for i, example in enumerate(raw_dataset[split]):
                problem_id = f"{split}_{i}"
                question = example["question"]
                answer = example["answer"]
                problem_splits[split].append(
                    ProblemRecord(id=problem_id, question=question, answer=answer)
                )
        return cls(
            problem_splits=problem_splits,
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            revision=revision,
        )


class Dataset:
    """A collection of problems with stable IDs, questions, and reference answers."""

    def __init__(
        self,
        problem_splits: dict[str, list[ProblemRecord]],
        dataset_id: str | None = None,
        dataset_config: str | None = None,
        revision: str | None = None,
        split_names: tuple[str, ...] | None = None,
    ) -> None:
        self.problem_splits = problem_splits
        self.dataset_id = dataset_id
        self.dataset_config = dataset_config
        self.revision = revision
        self.split_names = split_names

    @property
    def metadata(self) -> DatasetMetadata:
        """Describe the source and exact records currently held by this dataset."""
        effective_split_names = tuple(self.split_names) if self.split_names is not None else tuple(self.problem_splits.keys())
        return DatasetMetadata(
            dataset_id=self.dataset_id,
            dataset_config=self.dataset_config,
            revision=self.revision,
            split_names=effective_split_names,
            selected_ids=tuple(self.problem_splits[split][i].id for split in effective_split_names for i in range(len(self.problem_splits[split]))),
        )

    @classmethod
    def from_dataset_id(
        cls,
        dataset_id: str,
        split_option: str | None = None,
        dataset_config: str | None = None,
        revision: str | None = None,
    ) -> "Dataset":
        """Load a Hugging Face dataset, optionally selecting a split and revision."""
        loaded_dataset = LoadedDataset.load_from_huggingface(
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            revision=revision,
        )
        return cls.from_loaded_dataset(loaded_dataset, split_option)

    @classmethod
    def from_loaded_dataset(
        cls,
        loaded_dataset: LoadedDataset,
        split_option: str | None = None,
    ) -> "Dataset":
        """Create a Dataset from a LoadedDataset, optionally filtering by a specific split."""
        if split_option is not None:
            if split_option not in loaded_dataset.problem_splits:
                raise ValueError(
                    f"Invalid split option: {split_option}. Available splits: {list(loaded_dataset.problem_splits.keys())}"
                )
            problem_splits = {split_option: loaded_dataset.problem_splits[split_option]}
            split_names = (split_option,)
        else:
            problem_splits = loaded_dataset.problem_splits
            split_names = tuple(loaded_dataset.problem_splits.keys())
        return cls(
            problem_splits=problem_splits,
            dataset_id=loaded_dataset.dataset_id,
            dataset_config=loaded_dataset.dataset_config,
            revision=loaded_dataset.revision,
            split_names=split_names,
        )

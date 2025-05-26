from abc import ABC, abstractmethod
from typing import Literal
from dataclasses import dataclass


@dataclass
class MessageDTO:
    role: Literal['system', 'human']
    text: str


@dataclass
class QuestionDTO:
    text: str
    history: list[MessageDTO]


@dataclass
class AnswerDTO:
    text: str
    used_tokens: int


class LLMService(ABC):

    @abstractmethod
    def execute(self, data: QuestionDTO) -> AnswerDTO:
        raise NotImplemented

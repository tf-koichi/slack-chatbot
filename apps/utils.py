from typing import Optional, Any, Callable, Generator
import os
import re
import openai
from openai.error import InvalidRequestError
import tiktoken
from tenacity import retry, retry_if_not_exception_type, wait_fixed

class Messages:
    def __init__(self, tokens_estimator: Callable[[dict], int]) -> None:
        """Initializes the Messages class.
        Args:
            tokens_estimator (Callable[[Dict], int]):
                Function to estimate the number of tokens of a message.
                Args:
                    message (Dict): The message to estimate the number of tokens of.
                Returns:
                    (int): The estimated number of tokens.
        """
        self.tokens_estimator = tokens_estimator
        self.messages = list()
        self.num_tokens = list()
    
    def append(self, message: dict[str, str], num_tokens: Optional[int]=None) -> None:
        """Appends a message to the messages.
        Args:
            message (Dict[str, str]): The message to append.
            num_tokens (Optional[int]):
                The number of tokens of the message.
                If None, self.tokens_estimator will be used.
        """
        self.messages.append(message)
        if num_tokens is None:
            self.num_tokens.append(self.tokens_estimator(message))
        else:
            self.num_tokens.append(num_tokens)
    
    def trim(self, max_num_tokens: int) -> None:
        """Trims the messages to max_num_tokens."""
        while sum(self.num_tokens) > max_num_tokens:
            _ = self.messages.pop(1)
            _ = self.num_tokens.pop(1)
    
    def rollback(self, n: int) -> None:
        """Rolls back the messages by n steps."""
        for _ in range(n):
            _ = self.messages.pop()
            _ = self.num_tokens.pop()

class ChatEngine:
    """Chatbot engine that uses OpenAI's API to generate responses."""
    size_pattern = re.compile(r"\-(\d+)k")

    @classmethod
    def get_max_num_tokens(cls) -> int:
        """Returns the maximum number of tokens allowed for the model."""
        mo = cls.size_pattern.search(cls.model)
        if mo:
            return int(mo.group(1))*1024
        elif cls.model.startswith("gpt-3.5"):
            return 4*1024
        elif cls.model.startswith("gpt-4"):
            return 8*1024
        else:
            raise ValueError(f"Unknown model: {cls.model}")

    @classmethod
    def setup(cls, model: str, tokens_haircut: float|tuple[float]=0.9) -> None:
        """Basic setup of the class.
        Args:
            model (str): The name of the OpenAI model to use, i.e. "gpt-3-0613" or "gpt-4-0613"
            tokens_haircut (float|Tuple[float]): coefficients to modify the maximum number of tokens allowed for the model.
        """
        cls.model = model
        cls.enc = tiktoken.encoding_for_model(model)
        match tokens_haircut:
            case tuple(x) if len(x) == 2:
                cls.max_num_tokens = round(cls.get_max_num_tokens()*x[1] + x[0])
            case float(x):
                cls.max_num_tokens = round(cls.get_max_num_tokens()*x)
        
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @classmethod
    def estimate_num_tokens(cls, message: dict) -> int:
        """Estimates the number of tokens of a message.
        Args:
            message (Dict): The message to estimate the number of tokens of.
        Returns:
            (int): The estimated number of tokens.
        """
        return len(cls.enc.encode(message["content"]))
    
    def __init__(self) -> None:
        """Initializes the chatbot engine.
        """
        self.messages = Messages(self.estimate_num_tokens)
        self.messages.append({
            "role": "system",
            "content": "ユーザーを助けるチャットボットです。博多弁で答えます。"
        })
        self.completion_tokens_prev = 0
        self.total_tokens_prev = self.messages.num_tokens[-1]

    @retry(retry=retry_if_not_exception_type(InvalidRequestError), wait=wait_fixed(10))
    def _process_chat_completion(self, **kwargs) -> dict[str, Any]:
        """Processes ChatGPT API calling."""
        self.messages.trim(self.max_num_tokens)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages.messages,
            **kwargs
        )
        assert isinstance(response, dict)
        message = response["choices"][0]["message"]
        usage = response["usage"]
        self.messages.append(message, num_tokens=usage["completion_tokens"] - self.completion_tokens_prev)
        self.messages.num_tokens[-2] = usage["prompt_tokens"] - self.total_tokens_prev
        self.completion_tokens_prev = usage["completion_tokens"]
        self.total_tokens_prev = usage["total_tokens"]
        return message
    
    def reply_message(self, user_message: str) -> Generator:
        """Replies to the user's message.
        Args:
            user_message (str): The user's message.
        Yields:
            (str): The chatbot's response(s)
        """
        message = {"role": "user", "content": user_message}
        self.messages.append(message)
        try:
            message = self._process_chat_completion()
        except InvalidRequestError as e:
            yield f"## Error while Chat GPT API calling with the user message: {e}"
            return
        
        yield message['content']


from typing import Optional, Any, Generator
import os
import openai
from openai.error import InvalidRequestError
from tenacity import retry, retry_if_not_exception_type, wait_fixed

class ChatEngine:
    """Chatbot engine that uses OpenAI's API to generate responses."""
    @classmethod
    def setup(cls, model: str) -> None:
        """Basic setup of the class.
        Args:
            model (str): The name of the OpenAI model to use, i.e. "gpt-3-0613" or "gpt-4-0613"
        """
        cls.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    
    def __init__(self) -> None:
        """Initializes the chatbot engine.
        """
        self.messages = [{
            "role": "system",
            "content": "ユーザーを助けるチャットボットです。博多弁で答えます。"
        }]

    @retry(retry=retry_if_not_exception_type(InvalidRequestError), wait=wait_fixed(10))
    def _process_chat_completion(self, **kwargs) -> dict[str, Any]:
        """Processes ChatGPT API calling."""
        response = openai.ChatCompletion.create(model=self.model, messages=self.messages, **kwargs)
        assert isinstance(response, dict)
        message = response["choices"][0]["message"]
        self.messages.append(message)
        return message
    
    def reply_message(self, user_message: str) -> Generator:
        """Replies to the user's message.
        Args:
            user_message (str): The user's message.
        Yields:
            (str): The chatbot's response(s)
        """
        self.messages.append({"role": "user", "content": user_message})
        try:
            message = self._process_chat_completion()
        except InvalidRequestError as e:
            yield f"## Error while Chat GPT API calling with the user message: {e}"
            return
        
        yield message['content']


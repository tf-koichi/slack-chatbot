from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import io
import os
import re
import json
from pathlib import Path
import sqlite3
import pandas as pd
import openai
from openai.error import InvalidRequestError
import tiktoken
from tenacity import retry, retry_if_not_exception_type, wait_fixed

class WSDatabase:
    data_path = Path("../data/world_stats.sqlite3")
    schema = [
        {
            "name": "country",
            "description": "国名" 
        },{
            "name": "country_code",
            "description": "国コード"
        },{
            "name": "average life expectancy at birth",
            "description": "平均寿命（年）"
        },{
            "name": "alcohol_consumption",
            "description": "一人当たりの年間アルコール消費量（リットル）"
        },{
            "name": "region",
            "description": "地域"
        },{
            "name": "gdp per capita",
            "description": "一人当たりのGDP（ドル）"
        }
    ]
    def __enter__(self):
        self.conn = sqlite3.connect(self.data_path)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    @classmethod
    def schema_str(cls):
        schema_df = pd.DataFrame.from_records(cls.schema)
        text_buffer = io.StringIO()
        schema_df.to_csv(text_buffer, index=False)
        text_buffer.seek(0)
        schema_csv = text_buffer.read()
        schema_csv = "table: world_stats\ncolumns:\n" + schema_csv
        return schema_csv
    
    def ask_database(self, query):
        """Function to query SQLite database with a provided SQL query."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            cols = [col[0] for col in cursor.description]
            results_df = pd.DataFrame(results, columns=cols)
            text_buffer = io.StringIO()
            results_df.to_csv(text_buffer, index=False)
            text_buffer.seek(0)
            results_csv = text_buffer.read()
        except Exception as e:
            results_csv = f"query failed with error: {e}"

        return results_csv

class Messages:
    def __init__(self, tokens_estimator: Callable[[Dict], int]) -> None:
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
    
    def append(self, message: Dict[str, str], num_tokens: Optional[int]=None) -> None:
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
    def setup(cls, model: str, tokens_haircut: float|Tuple[float]=0.9, quotify_fn: Callable[[str], str]=None) -> None:
        """Basic setup of the class.
        Args:
            model (str): The name of the OpenAI model to use, i.e. "gpt-3-0613" or "gpt-4-0613"
            tokens_haircut (float|Tuple[float]): coefficients to modify the maximum number of tokens allowed for the model.
            quotify_fn (Callable[[str], str]): Function to quotify a string.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        cls.model = model
        cls.enc = tiktoken.encoding_for_model(model)
        if isinstance(tokens_haircut, tuple):
            cls.max_num_tokens = round(cls.get_max_num_tokens()*tokens_haircut[1] + tokens_haircut[0])
        else:
            cls.max_num_tokens = round(cls.get_max_num_tokens()*tokens_haircut)

        cls.functions = [
            {
                "name": "ask_database",
                "description": "世界各国の平均寿命、アルコール消費量、一人あたりGDPのデータベースを検索するための関数。出力はSQLite3が理解できる完全なSQLクエリである必要がある。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                SQL query extracting info to answer the user's question.
                                SQL should be written using this database schema:
{WSDatabase.schema_str()}
                            """,
                        }
                    },
                    "required": ["query"]
                },
            }
        ]

        if quotify_fn is None:
            cls.quotify_fn = staticmethod(lambda x: x)
        else:
            cls.quotify_fn = staticmethod(quotify_fn)


    @classmethod
    def estimate_num_tokens(cls, message: Dict) -> int:
        """Estimates the number of tokens of a message.
        Args:
            message (Dict): The message to estimate the number of tokens of.
        Returns:
            (int): The estimated number of tokens.
        """
        return len(cls.enc.encode(message["content"]))
    
    def __init__(self, style: str="博多弁") -> None:
        """Initializes the chatbot engine.
        """
        style_direction = f"{style}で答えます" if style else ""
        self.style = style
        self.messages = Messages(self.estimate_num_tokens)
        self.messages.append({
            "role": "system",
            "content": f"必要に応じてデータベースを検索し、ユーザーを助けるチャットボットです。{style_direction}"
        })
        self.completion_tokens_prev = 0
        self.total_tokens_prev = self.messages.num_tokens[-1]
        self.verbose = True

    @retry(retry=retry_if_not_exception_type(InvalidRequestError), wait=wait_fixed(10))
    def _process_chat_completion(self, **kwargs) -> Dict[str, Any]:
        """Processes ChatGPT API calling."""
        self.messages.trim(self.max_num_tokens)
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages.messages,
            **kwargs
        )
        message = response["choices"][0]["message"]
        usage = response["usage"]
        self.messages.append(message, num_tokens=usage["completion_tokens"] - self.completion_tokens_prev)
        self.messages.num_tokens[-2] = usage["prompt_tokens"] - self.total_tokens_prev
        self.completion_tokens_prev = usage["completion_tokens"]
        self.total_tokens_prev = usage["total_tokens"]
        return message
    
    def reply_message(self, user_message: str) -> None:
        """Replies to the user's message.
        Args:
            user_message (str): The user's message.
        Yields:
            (str): The chatbot's response(s)
        """
        message = {"role": "user", "content": user_message}
        self.messages.append(message)
        try:
            message = self._process_chat_completion(
                functions=self.functions,
            )
        except InvalidRequestError as e:
            yield f"## Error while Chat GPT API calling with the user message: {e}"
            return
        
        while message.get("function_call"):
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            if self.verbose:
                yield self.quotify_fn(f"function name: {function_name}")
                yield self.quotify_fn(f"arguments: {arguments}")
            
            if function_name == "ask_database":
                with WSDatabase() as db:
                    function_response = db.ask_database(arguments["query"])
            else:
                function_response = f"## Unknown function name: {function_name}"

            if self.verbose:
                yield self.quotify_fn(f"function response:\n{function_response}")
                        
            self.messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_response
            })
            try:
                message = self._process_chat_completion()
            except InvalidRequestError as e:
                yield f"## Error while ChatGPT API calling with the function response: {e}"
                self.messages.rollback(3)
                return

        yield message['content']


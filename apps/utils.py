from email import message
from typing import Optional, Union, Any, Callable, Generator
import io
import os
import re
import shutil
import json
from json import JSONEncoder
from pathlib import Path
import traceback
from IPython.core.interactiveshell import InteractiveShell
import sqlite3
import numpy as np
import pandas as pd
import openai
from openai.error import InvalidRequestError
import tiktoken
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, wait_fixed

class NumpyArrayEncoder(JSONEncoder):
    """Custom JSONEncoder class to handle numpy arrays. cf. https://pynative.com/python-serialize-numpy-ndarray-into-json/"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class WSDatabase:
    data_path = Path(__file__).parent.joinpath("../data/world_stats.sqlite3")
    schema = [
        {
            "name": "country",
            "description": "国名" 
        },{
            "name": "country_code",
            "description": "国コード"
        },{
            "name": "average_life_expectancy_at_birth",
            "description": "平均寿命（年）"
        },{
            "name": "alcohol_consumption",
            "description": "一人当たりの年間アルコール消費量（リットル）"
        },{
            "name": "region",
            "description": "地域"
        },{
            "name": "gdp_per_capita",
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
    
    def query(self, query: str) -> pd.DataFrame:
        """Function to query SQLite database with a provided SQL query.
        Args:
            query (str): The SQL query to execute.
        Returns:
            (pd.DataFrame): The results of the query.
        """
        results_df = pd.read_sql_query(query, self.conn)
        return results_df

    def ask_database(self, query: str) -> str:
        """Function to query SQLite database with a provided SQL query.
        Args:
            query (str): The SQL query to execute.
        Returns:
            (str): The results of the query in CSV format.
        """
        results_df = self.query(query)
        text_buffer = io.StringIO()
        results_df.to_csv(text_buffer, index=False)
        text_buffer.seek(0)
        results_csv = text_buffer.read()
        return results_csv

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
    
    def rollback(self, n: Optional[int]=None) -> None:
        """Rolls back the messages.
        Args:
            n (Optional[int]): The number of messages to roll back.
                If None, roll back to the last user message.
        """
        if n is None:
            while True:
                message = self.messages.pop()
                _ = self.num_tokens.pop()
                if message["role"] == "user":
                    break
            
        else:
            for _ in range(n):
                _ = self.messages.pop()
                _ = self.num_tokens.pop()

class ChatEngine:
    """Chatbot engine that uses OpenAI's API to generate responses."""
    size_pattern = re.compile(r"\-(\d+)k")
    code_pattern = re.compile(r"code\s*=\s*([\"']+)(.*)\1", re.DOTALL|re.MULTILINE)
    graph_pattern = re.compile(r"^!\[(?:.+)\]\((?:.*?)([^/]+\.(?:png|jpg|jpeg))\)", re.MULTILINE)

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
    def setup(cls, model: str, tokens_haircut: float|tuple[float]=0.9, quotify_fn: Callable[[str], str]=lambda x: x) -> None:
        """Basic setup of the class.
        Args:
            model (str): The name of the OpenAI model to use, i.e. "gpt-3-0613" or "gpt-4-0613"
            tokens_haircut (float|Tuple[float]): coefficients to modify the maximum number of tokens allowed for the model.
            quotify_fn (Callable[[str], str]): Function to quotify a string.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        cls.model = model
        cls.enc = tiktoken.encoding_for_model(model)
        match tokens_haircut:
            case tuple(x) if len(x) == 2:
                cls.max_num_tokens = round(cls.get_max_num_tokens()*x[1] + x[0])
            case float(x):
                cls.max_num_tokens = round(cls.get_max_num_tokens()*x)
            case _:
                raise ValueError(f"Invalid tokens_haircut: {tokens_haircut}")

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
            }, {
                "name": "exec_python",
                "description": """
                    Use this function to execute Python code. It will return the result as json. The result contains the following items:
                    - "error_in_exec":
                    - "result": STDOUT and the value of the last expression.
                    - "files": Files in your sandbox.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": """
                                Python code to execute.
                                Notes:
                                - The following code has been executed before your code:
                                    ```Python
                                    import numpy as np
                                    import pandas as pd
                                    import matplotlib.pyplot as plt
                                    import seaborn as sns
                                    import sys
                                    sys.path.append('../')
                                    from utils import WSDatabase
                                    ```
                                - You can use WSDatabase to access the database as following:
                                    ```Python
                                    with WSDatabase() as db:
                                        result = db.query("SELECT * FROM world_stats WHERE country = 'Japan'") # result is a DataFrame of Pandas.
                                    ```
                                - You can use scikit-learn. You have to import it by yourself.
                                - Do not return json-unserializable objects such as Pandas DataFrame as the last expression. Numpy ndarray is OK.
                                - Use `/n` as a newline character, not `\r\n`.
                            """,
                        }
                    },
                    "required": ["code"]
                }
            }, {
                "name": "post_image",
                "description": "Use this function to post an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename of the image to post."
                        }
                    }
                }
            }
        ]

        cls.quotify_fn = staticmethod(quotify_fn)
        
        for sandbox in Path("./").glob("sandbox-*"):
            shutil.rmtree(sandbox)

    @classmethod
    def estimate_num_tokens(cls, message: dict) -> int:
        """Estimates the number of tokens of a message.
        Args:
            message (Dict): The message to estimate the number of tokens of.
        Returns:
            (int): The estimated number of tokens.
        """
        if "content" in message.keys():
            return len(cls.enc.encode(message["content"]))
        else:
            return 0
    
    def __init__(self, user_id: str, post_image: Callable, style: str="博多弁") -> None:
        """Initializes the chatbot engine.
        """
        style_direction = f"{style}で答えます" if style else ""
        self.style = style
        self.messages = Messages(self.estimate_num_tokens)
        self.messages.append({
            "role": "system",
            "content": f"""
                必要に応じてデータベースを検索たりPythonコードを作成・実行して、ユーザーを助けるチャットボットです。{style_direction}
                I have a sandbox directory to execute Python code. I can post an image in the directory with markdown style, i.e. `![alt](filename)`
            """
        })
        self.completion_tokens_prev = 0
        self.total_tokens_prev = self.messages.num_tokens[-1]
        self._verbose = True
        self.sandbox_dir = Path("./sandbox-" + user_id).absolute()
        self.sandbox_dir.mkdir(exist_ok=True)
        self.python_shell = InteractiveShell(ipython_dir=str(self.sandbox_dir))
        self.python_shell.run_cell(f"""
import os
os.chdir('{self.sandbox_dir}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import sys
sys.path.append('../')
from utils import WSDatabase
""")
        self.post_image = post_image

    @property
    def verbose(self) -> bool:
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value
    
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
        message = self._process_chat_completion(
            functions=self.functions,
        )                
        while message.get("function_call"):
            function_name = message["function_call"]["name"]
            try:
                arguments = json.loads(message["function_call"]["arguments"])
            except json.decoder.JSONDecodeError as e:
                if function_name == "exec_python":
                    yield self.quotify_fn(f"## JSONDecodeError.")
                    mo = self.code_pattern.search(message["function_call"]["arguments"])
                    if mo:
                        yield self.quotify_fn(f"## Interpret as `code = ` pattern.")
                        arguments = {"code": mo.group(2)}
                    else:
                        yield self.quotify_fn(f"## Interpret as a code literal.")
                        arguments = {"code": message["function_call"]["arguments"]}
                else:
                    yield self.quotify_fn(f"## JSONDecodeError.")
                    self.messages.rollback()
                    return
        
            if self._verbose:
                yield self.quotify_fn(f"function name: {function_name}")
                yield self.quotify_fn(f"arguments:\n{arguments}")
            
            try:
                if function_name == "ask_database":
                    with WSDatabase() as db:
                        function_response = db.ask_database(arguments["query"])
                elif function_name == "exec_python":
                    function_response = self.exec_python(arguments["code"])
                elif function_name == "post_image":
                    function_response = self.post_image(filename=self.sandbox_dir / arguments["filename"])
                else:
                    function_response = f"## Unknown function name: {function_name}"
            except Exception as e:
                function_response = f"## Error while executing the function:\n{type(e).__name__}: {str(e)}"

            if self._verbose:
                yield self.quotify_fn(f"function response:\n{function_response}")
                        
            self.messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": function_response
            })
            message = self._process_chat_completion()

        b = 0
        for mo in self.graph_pattern.finditer(message["content"]):
            yield message["content"][b: mo.start()]
            self.post_image(filename=self.sandbox_dir / mo.group(1))
            b = mo.end()

        yield message["content"][b:]

    def exec_python(self, code: str) -> str:
        """Executes Python code.
        Args:
            code (str): The Python code to execute.
        Returns:
            (str): The result of the execution.
        """
        result = self.python_shell.run_cell(code)
        files_in_sandbox = [f.name for f in self.sandbox_dir.glob("*") if f.is_file()]
        result_json = json.dumps({
            "error_in_exec": result.error_in_exec,
            "result": result.result,
            "files": files_in_sandbox
        }, cls=NumpyArrayEncoder)
        return result_json

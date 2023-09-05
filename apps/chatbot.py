import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from utils import ChatEngine

chatbot_app_token = os.environ["CHATBOT_APP_TOKEN"]
slack_bot_token = os.environ["SLACK_BOT_TOKEN"]

app = App(token=slack_bot_token)

@app.message()
def handle(message, say):
    global chat_engine_dict
    if message["user"] not in chat_engine_dict.keys():
        chat_engine_dict[message["user"]] = ChatEngine()
    
    for reply in chat_engine_dict[message["user"]].reply_message(message['text']):
        say(reply)

model = "gpt-4-0613"
ChatEngine.setup(model)
chat_engine_dict = dict()
SocketModeHandler(app, chatbot_app_token).start()

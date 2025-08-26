import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from collections import deque
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema import HumanMessage, AIMessage

# Load API key
load_dotenv()
llm = ChatOpenAI(model="gpt-5-nano", temperature=1 ,openai_api_key=os.getenv("OPENAI_API_KEY"))
def get_history_text(player_id):
    history = get_session_history(player_id)
    return "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
prompt_template = """
You are an NPC in a fantasy video game.
You have a personality and mood that can change based on the player's interactions.

Recent player messages (oldest to newest):
{history}

Latest player message:
"{question}"

Instructions:
- Reply as the NPC in 1–2 sentences.
- Determine your mood based on the player's messages. Choose one of: Neutral, Angry, Friendly.
- Reply naturally in character.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=[ "history", "question"])

class NPC_RESPONSE(BaseModel):
    mood: Literal["Neutral","Angry","Friendly"] = Field(description="Your task is to determine the NPC's mood based on the player's message and recent history. Choose from Neutral , Angry, or Friendly.")
    reply : str = Field(description="The NPC's reply to the player")
define_llm = llm.with_structured_output(NPC_RESPONSE)

chat_store = {}
class InMemoryChatHistory(BaseChatMessageHistory):
    """Custom chat history implementation using in-memory storage"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        if session_id not in chat_store:
            chat_store[session_id] = deque(maxlen=6)
        self.messages = chat_store[session_id]

    def add_chat(self, message, role='user'):
        if role == "user":
            self.messages.append(HumanMessage(content=message))
        else:
            self.messages.append(AIMessage(content=message))
        return message

    def clear(self):
        """Clear the chat history"""
        self.messages.clear()
        if self.session_id in chat_store:
            del chat_store[self.session_id]

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat history for a session"""
    return InMemoryChatHistory(session_id)

def npc_response(question,player_id):
    chain = prompt | define_llm 
    history = get_session_history(player_id)
    npc_reply = chain.invoke({"question": question,"history":get_history_text(player_id)})
    history.add_chat(question,role="user")
    history.add_chat(npc_reply.reply,role="assistant")
    return npc_reply



import json
from datetime import datetime

if __name__ == "__main__":
    
    with open("players.json", "r") as file:
        data = json.load(file)
    sorted_data = sorted(data, key=lambda x: datetime.fromisoformat(x["timestamp"]))
    log_file = open("npc_log.txt", "w")
    player = {}
    counter=1
    for entry in sorted_data:
        player_id = entry["player_id"]
        player_message = entry["text"]
        npc_res = npc_response(player_message,player_id)
        print(f"({counter})")
        log_file.write(f"Conversation {counter}:\n")
        log_file.write(f"Player ({player_id}): {player_message}\n")
        log_file.write(f"Latest Message : {player_message}\n")
        log_file.write(f"NPC Reply: {npc_res.reply}\n")
        if player_id not in player:
            player[player_id] = deque(maxlen=3)
            player[player_id].append("Neutral")        
        else:
            player[player_id].append(npc_res.mood)
        log_file.write(f"NPC Mood: {player[player_id][-1]}\n")
        log_file.write(f"Last 3 Message : {[msg.content for msg in get_session_history(player_id).messages if msg.type=='human']}\n")
        log_file.write(f"Last 3 Message NPC Mood: {[mood for mood in player[player_id]]}\n")
        log_file.write(f"Timestamp: {entry['timestamp']}\n")
        log_file.write("-" * 40 + "\n")
        counter+=1


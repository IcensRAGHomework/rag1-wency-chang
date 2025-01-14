import json
import traceback
import re
import os
import openai

from operator import itemgetter
from typing import List

from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI
from langchain_core.messages import HumanMessage
from model_configurations import get_model_configuration
from calendarific_api import get_calendar_agent
from image_handle import local_image_to_data_url
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from mimetypes import guess_type

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    try:
        formatted_question = (
            f"{question}\n\n"
            "請只返回 JSON 格式的內容，並且不包含任何額外的文字或格式，例如 Markdown。\n"
            "請以以下JSON格式回答：\n"
            "{"
            '    "Result": ['
            '        {'
            '            "date": "YYYY-MM-DD",'
            '            "name": "紀念日名稱"'
            '        }'
            "    ]"
            "}"
        )
        response = demo(formatted_question)
        return getJsonMatch(response.content)
    except json.JSONDecodeError:
        return {"Result": response.content}
    except Exception as e:
        return {"Error": str(e), "Traceback": traceback.format_exc()}

def generate_hw02(question):
    session_id = "hw2"
    agent = get_calendar_agent()
    result = agent.invoke({"input": question})
    output = result['output']
    json_result = getJsonMatch(output)
    return json_result

def generate_hw03(question2, question3):
    session_id = "hw3"
    agent_with_history = RunnableWithMessageHistory(
        get_calendar_agent(), 
        get_session_history=get_by_session_id,
        history_messages_key="history"
    )
    agent_with_history.invoke({"input": question2}, config={"configurable": {"session_id": session_id}})
    result = agent_with_history.invoke({"input": f"{question3} 如果不在清單請回答應該增加與增加原因，反之亦然"}, config={"configurable": {"session_id": session_id}})
    json_result = getJsonMatch(result['output'])
    return json_result

def getJsonMatch(result):
    json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
    if json_match:
        json_data = json_match.group(1)
        data = json.loads(json_data)
        json_result = {"Result":data}        
    else:
        json_result = {"Result": json.loads(result)}
    string_result = json.dumps(json_result, indent=4, ensure_ascii=False)
    return string_result
    
def generate_hw04(question):
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    
    client = AzureOpenAI(
        api_key=gpt_config['api_key'],  
        api_version=gpt_config['api_version'],
        base_url=f"{gpt_config['api_base']}openai/deployments/{gpt_config['deployment_name']}",
    )
    response = client.chat.completions.create(
        model=gpt_config['deployment_name'],
        messages=[
            { "role": "system", "content": "You are a helpful assistant. please answer the question with json format: score: ?" },
            { "role": "user", "content": [  
                { 
                    "type": "text", 
                    "text": f"{question}:" 
                },
                { 
                    "type": "image_url",
                    "image_url": {
                        "url": f"{data_url}"
                    }
                }
            ] } 
        ],
        max_tokens=2000 
    )
    content = response
    if response and response.choices:
        content = response.choices[0].message.content
    result = getJsonMatch(content)
    return result
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    return response
    
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

import requests
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor, tool, AgentType
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from model_configurations import get_model_configuration
from pydantic import BaseModel


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

API_KEY = 'DThnDgXCpyDAe2b5gqd8TyrwZqrFo3Ej'    
BASE_URL = 'https://calendarific.com/api/v2/holidays'

class CalendarArgsSchema(BaseModel):
    country: str
    year: int
    month: int

def query_calendarific_api(action_input: CalendarArgsSchema):
    country = action_input.country
    year = action_input.year
    month = action_input.month
    import requests
    params = {
        'api_key': API_KEY,
        'country': country,
        'year': year,
        'month': month,
        'language': 'zh'
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        holidays = response.json()['response']['holidays']
        return [{'date': holiday['date']['iso'], 'name': holiday['name']} for holiday in holidays]
    else:
        return {'error': 'Failed to retrieve holidays'}


llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

calendar_tool = StructuredTool.from_function(
    name="query_calendarific_api",
    func=query_calendarific_api,
    description="Get holidays for a specific year and month from the Calendarific API.",
    
)

prompt = ChatPromptTemplate.from_messages(
	[
		("system", "You are a helpful assistant who has the ability to verify the holidays"),
		("human", "{input}"),
        ("assistant", "The expected format for your answer is: json array with dates and holidays' name, or json: add: true/false, reson: why, and plesase no other content/info except the array"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
	]
)

agent = create_tool_calling_agent(
	llm=llm,
	prompt=prompt,
    tools=[calendar_tool],
)

agent_executor = AgentExecutor(
	agent=agent,
	tools=[calendar_tool],
	verbose=True
)

def get_calendar_agent():
    return agent_executor

from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

# memory_key is the name of the additional variable added by memory
# return_messages is set to true because we are working with a conversational LLM
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
                            messages=[MessagesPlaceholder(variable_name="messages"),HumanMessagePromptTemplate.from_template("{content}")])

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory
)

while True:
    content = input('>>> ')
    result = chain({"content": content})
    print(result)
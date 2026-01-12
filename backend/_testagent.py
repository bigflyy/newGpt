from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Create the actual function separately
def _get_weather(location: str) -> str:
    """Get weather for a location"""
    return f"Sunny in {location}"

# Create the tool from the function
get_weather_tool = tool(_get_weather)

# Initialize model
model = ChatOllama(model="qwen3:8b", temperature=0)
model_with_tools = model.bind_tools([get_weather_tool])

# Step 1: First call - model decides to call tool
initial_response = model_with_tools.invoke([
    HumanMessage(content="What's the weather in New York?")
])

print("Model wants to call tool:", initial_response.tool_calls)

# Step 2: Execute the tool function
tool_call = initial_response.tool_calls[0]
tool_result = get_weather_tool.invoke(tool_call['args'])
print("Tool result:", tool_result)

# Step 3: Send tool result back to model
final_response = model.invoke([
    HumanMessage(content="What's the weather in New York?"),
    ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
])

print("Final answer:", final_response.content)
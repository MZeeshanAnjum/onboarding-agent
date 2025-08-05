from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompts.prompt import *
from pydantic import BaseModel, Field
from typing import Optional, List, Annotated
from llm import llm
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
import logging
from logging import getLogger

import base64
from io import BytesIO

logger = getLogger("onboarding_agent")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def override(_, new):
    return new

class CheckRequirementsResponse(BaseModel):
    user_confirmation: bool = False
    response: str = ""

class VerifyInformation(BaseModel):
    satisfied: bool = False

class ToolInfo(BaseModel):
    tools_needed: List[str] = Field(default_factory=list)
    tools_suggested: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    query: Optional[str] = None
    context_history: Annotated[list[dict], override] =[]
    get_tools_flag: bool = True
    has_enough_information: bool = False
    user_confirmation: bool = False
    tools: ToolInfo = Field(default_factory=ToolInfo)
    tools_selected: List[str] = []
    response: str= ""
    intent: str = ""
    summary: str = ""
    available_tools: dict = {}



class OnboardingAgent:
    def __init__(self):
        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("GatherInformation", self.gather_information)
        builder.add_node("VerifyInformation", self.verify_information)
        builder.add_node("IntentClassifier", self.intent_classifier)
        builder.add_node("GetTools", self.get_tools_node)
        builder.add_node("CheckRequirements", self.check_requirements_node)
        builder.add_node("SuggestTool", self.tools_needed_node)
        builder.add_node("GenerateSummary", self.generate_summary)

        builder.add_conditional_edges(START, self.route_start, {"IntentClassifier": "IntentClassifier", "GatherInformation": "GatherInformation"})
        builder.add_edge("GatherInformation", "VerifyInformation")
        builder.add_conditional_edges("VerifyInformation", self.check_tool_fetch, {"GetTools": "GetTools", "SuggestTool": "SuggestTool", "GenerateSummary": "GenerateSummary"})
        builder.add_conditional_edges("IntentClassifier", self.route_intent_classifier, {"GetTools": "GetTools", "SuggestTool": "SuggestTool", "GatherInformation": "GatherInformation"})
        builder.add_edge("GetTools", "SuggestTool")
        builder.add_edge("SuggestTool", "CheckRequirements")
        builder.add_edge("CheckRequirements", "GenerateSummary")
        builder.add_edge("GenerateSummary", END)

        return builder.compile()
    
    def route_start(self, state: AgentState) -> str:

        if state.has_enough_information:
            return "IntentClassifier"
        else:
            return "GatherInformation"

    def route_intent_classifier(self, state: AgentState) -> str:
        if state.has_enough_information and state.get_tools_flag:
            return "GetTools"
        elif state.has_enough_information and not state.get_tools_flag:
            return "SuggestTool"

    def check_tool_fetch(self,state: AgentState) -> str:
        logger.info("In check_tool_fetch")
        
        if state.get_tools_flag and state.has_enough_information:
            logger.info("Fetching tools")
            return "GetTools"
        elif not state.get_tools_flag and state.has_enough_information:
            print("Skipping tool fetch")
            logger.info("Skipping tool fetch")
            return "SuggestTool"
        elif not state.has_enough_information:
            logger.info("Not enough information, going back to gather information")
            return "GenerateSummary"

    def intent_classifier(self, state: AgentState) -> Command:
        print("Classifying intent")

        prompt = INTENT_CLASSIFIER_PROMPT.format(
            query=state.query,
            context_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state.context_history] if state.context_history else ""),
            )

        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)
        print(f"\n Response: {response.content}")

        if response.content == "RoleModification":
            return Command(
                goto = "GatherInformation")

    def gather_information(self, state: AgentState) -> AgentState:
        logger.info("Gathering information")
        print("Gathering information")

        prompt=ASK_FOLLOWUP_QUESTION_PROMPT.format(
            query=state.query,
            context_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state.context_history] if state.context_history else ""),
        )
        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)
        print(f"\n Response: {response}")
        response = response.content

        state.context_history.append({"role": "user", "content": state.query})
        state.context_history.append({"role": "assistant", "content": response})

        print(f"\n Updated context history: {state.context_history}")

        return {"response": response, "context_history": state.context_history}

    def verify_information(self,state: AgentState) -> AgentState:
        print("Verifying information")

        prompt=VERIFY_INFORMATION_PROMPT.format(
            context_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state.context_history] if state.context_history else ""),
        )

        messages = [AIMessage(content=prompt)]
        structured_llm = llm.with_structured_output(VerifyInformation)
        response = structured_llm.invoke(messages)
        print(f"\n Response: {response}")
        if response.satisfied:
            context_history = state.context_history[:-1]
        else:  # Exclude the last user message
            context_history = state.context_history
        return {"has_enough_information": response.satisfied,"context_history": context_history}

    def get_tools_node(self,state: AgentState) -> AgentState:

        tools = {"list_repositories": "To list Github repos","create_repository" :"To create Github repo", "delete_repository" :"To delete github repo", "send_outlook_email" :"To send Outlook email","send_email":"To send mail on gmail","search_emails":"To send mail on gmail","delete_email":"Delete a mail on gmail","list_drive_files":"List files of the Google drive", "search_drive_files":"Search a file on google drive", "read_drive_file":"Read a file on google drive","delete_drive_file":"Delete a file on Google Drive"} # Simulated tool retrieval
        print("Retrieving tools") #TODO
        message={"role": "system", "content": f"You have access to the following tools: {tools}"}
        context_history=state.context_history + [message]
        return {"get_tools_flag": False, "available_tools": tools}

    def tools_needed_node(self,state: AgentState) -> AgentState:
        print("Suggesting Tools")

        context_history = state.context_history
        user_query = state.query
        print(state.tools.tools_needed)
        formatted_context_history = ""
        if context_history:
            for message in context_history:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"

        context_history = context_history + [{"role": "user", "content": user_query}]
        prompt= QUERY_PROMPT.format(
            query=user_query,
            context_history=formatted_context_history,
            tools=state.tools_selected,
            available_tools=state.available_tools
        )

        structured_llm =llm.with_structured_output(ToolInfo)
        messages = [AIMessage(content=prompt)]
        response=structured_llm.invoke(messages)
        print(response)
        if state.tools_selected:
            for tool in state.tools_selected:
                if tool not in response.tools_needed:
                    response.tools_needed.append(tool)

        return {"tools": ToolInfo(tools_needed=response.tools_needed, tools_suggested=response.tools_suggested), "context_history": context_history}

    def check_requirements_node(self,state: AgentState) -> AgentState:
        print("In check_requirements_node")


        formatted_context_history = ""
        if state.context_history:
            for message in state.context_history:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"

        tools_needed = [tool.replace("_", " ") for tool in state.tools.tools_needed if tool]
        tools_suggested = [tool.replace("_", " ") for tool in state.tools.tools_suggested if tool]
        print(f"\n Tools needed: {tools_needed}")
        print(f"\n Tools suggested: {tools_suggested}")

        prompt= Check_Requirement_PROMPT.format(
            query=state.query,
            tools_needed=tools_needed,
            tools_suggested=tools_suggested,
            context_history= formatted_context_history,
            available_tools=state.available_tools)

        messages = [AIMessage(content=prompt)]
        structured_llm = llm.with_structured_output(CheckRequirementsResponse)
        response=structured_llm.invoke(messages)
        print(f"\n {response.response}")
        context_history = state.context_history + [{"role": "assistant", "content": response.response}]
        return {"context_history": context_history, "response": response.response}

    def generate_summary(self, state: AgentState) -> str:

        formatted_context_history = ""
        if state.context_history:
            for message in state.context_history:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"
        
        prompt = GENERATE_SUMMARY_PROMPT.format(
            query=state.query,
            context_history=formatted_context_history,
            tools_needed=state.tools.tools_needed,
            tools_suggested=state.tools.tools_suggested
        )
        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)
        print(f"\n Summary: {response.content}")
        return {"summary": response.content}

    def user_confirmation(self, state: AgentState) -> AgentState:
        print("In User confirmation")

        prompt = USER_CONFIRMATION_PROMPT.format(
            summary=state.summary,
            tools_needed=state.tools.tools_needed,
            tools_suggested=state.tools.tools_suggested
        )
        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)
        print(f"\n User Confirmation: {response.content}")
        return {"response": response.content}

    def visualize(self):
        logger.info("Getting visualized planner agent")
        png_image   = self.graph.get_graph(xray=True).draw_mermaid_png()
        image_buf   = BytesIO(png_image)
        img_str     = base64.b64encode(image_buf.getvalue()).decode("utf-8")
        return {"selection_agent_base64": f"data:image/png;base64,{img_str}"}

    def check_satisfaction(self, state: AgentState) -> bool:
        logger.info("Checking satisfaction")
        if state.satisfactory:
            return "user_confirmation"
        return END
    
if __name__ == "__main__":
    agent = OnboardingAgent()
    # Example usage
    input_state = {
        "query": "I want to delete a file on github and send a mail to the client.?",
        "context_history": [{'role': 'system', 'content': "You have access to the following tools: ['github', 'gmail', 'outlook']"}],
        "get_tools_flag": False,
    }
    image_dict = agent.visualize()
    with open("graph.png", "wb") as f:
        f.write(base64.b64decode(image_dict["selection_agent_base64"].split(",")[1]))
    #result = agent.graph.invoke(input_state)
    #print(f"\n Final Result: {result}")
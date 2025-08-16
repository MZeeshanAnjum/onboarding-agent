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

from agents_helper.ag2 import RAGAgent
from utils.rag import Initialize_vector_store

import base64
from io import BytesIO

import json

from collections import defaultdict

logger = getLogger("onboarding_agent")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open('topics.json', 'r', encoding='utf-8') as f:
    categories = json.load(f)

def get_subtopics(category_name):
    subtopics = categories.get(category_name)
    if subtopics is not None:
        return subtopics
    else:
        return f"No such category '{category_name}' found."


def override(_, new):
    return new

class BusinessInfoChecklist(BaseModel):
    business_overview: bool = False
    industry_domain: bool = False
    target_market_customers: bool = False
    business_model: bool = False
    products_or_services: bool = False
    teams_departments_culture: bool = False
    technology_stack_tools: bool = False
    marketing_branding: bool = False
    customer_support_service: bool = False
    supply_chain_operations: bool = False
    financials_performance: bool = False
    current_challenges_growth_focus: bool = False
    legal_compliance: bool = False
    risk_management_security: bool = False
    innovation_research: bool = False
    strategic_partnerships_alliances: bool = False
    corporate_governance: bool = False
    environmental_social_governance_esg: bool = False
    knowledge_management_training: bool = False
    customer_experience_journey: bool = False
    international_business_considerations: bool = False
    exit_strategy_succession_planning: bool = False

class OnboardingResponse(BaseModel):
    intro: bool = False
    response: str = ""


class UserConfirmation(BaseModel):
    user_confirmation: bool=False
    response:str = ""

class AskUnansweredQuestions(BaseModel):

    all_questions_answered: bool=False
    response:str = ""

class AgentState(BaseModel):
    documents_uploaded:bool = False
    RAG:bool = False
    RAG_summary: str = ""
    unanswered_questions: List[dict] = []
    query: str = ""
    context_history: Annotated[list[dict], override] =[]
    summary: str = ""
    data: BusinessInfoChecklist = BusinessInfoChecklist()
    user_confirmation: bool = False
    suggested_agents: str = ""
    all_questions_answered: bool = False
    question_index: int = 0
    intro:bool = False


class OnboardingAgent:
    def __init__(self):
        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("GatherInformation", self.gather_information)
        builder.add_node("VerifyInformation", self.verify_information)
        builder.add_node("GenerateSummary", self.generate_summary)
        builder.add_node("UserConfirmation", self.user_confirmation)
        builder.add_node("SuggestAgents", self.suggest_agents)
        builder.add_node("RAGBasedAgent", self.rag_based_agent)
        builder.add_node("AskUnansweredQuestions", self.ask_unanswered_questions)

        # builder.add_conditional_edges(START, self.route, {"GatherInformation": "GatherInformation", "UserConfirmation": "UserConfirmation"})
        builder.add_conditional_edges(START,self.check_documents_uploaded, {"UserConfirmation": "UserConfirmation", "VerifyInformation": "VerifyInformation", "AskUnansweredQuestions": "AskUnansweredQuestions"})
        builder.add_conditional_edges("VerifyInformation", self.route, {"UserConfirmation": "UserConfirmation", "GatherInformation": "GatherInformation","AskUnansweredQuestions": "AskUnansweredQuestions"})
        builder.add_edge("UserConfirmation", "GenerateSummary")
        builder.add_edge("GatherInformation", "GenerateSummary")
        builder.add_edge("RAGBasedAgent", "AskUnansweredQuestions")
        builder.add_edge("AskUnansweredQuestions","GenerateSummary")
        builder.add_conditional_edges("GenerateSummary",self.confirmation_route, {"SuggestAgents": "SuggestAgents", "END": END})
        builder.add_conditional_edges("AskUnansweredQuestions",self.check_all_questions_answered, {"UserConfirmation": "UserConfirmation", "END": END})
        builder.add_edge("GenerateSummary", END)
        # builder.add_edge(START, "VerifyInformation")
        # builder.add_conditional_edges("VerifyInformation", self.route, {"UserConfirmation": "UserConfirmation", "GatherInformation": "GatherInformation"})
        # builder.add_edge("UserConfirmation", "GenerateSummary")
        # builder.add_edge("GatherInformation", "GenerateSummary")
        # builder.add_conditional_edges("GenerateSummary",self.confirmation_route, {"SuggestAgents": "SuggestAgents", "END": END})

        return builder.compile()

    def check_documents_uploaded(self, state: AgentState) -> str:
        if state.documents_uploaded and state.all_questions_answered:
            return "UserConfirmation"
        elif state.documents_uploaded and not state.all_questions_answered:
            return "AskUnansweredQuestions"
        else:
            return "VerifyInformation"

    def check_all_questions_answered(self, state: AgentState) -> str:
        if state.all_questions_answered:
            return "UserConfirmation"
        else:
            return "END"

    def route(self, state: AgentState) -> str:

        if all(state.data.dict().values()):
            return "UserConfirmation"
        else:
            return "GatherInformation"
    
    def confirmation_route(self, state: AgentState) -> str:
        if state.user_confirmation:
            return "SuggestAgents"
        else:
            return "END"

    def gather_information(self, state: AgentState) -> AgentState:
        logger.info("Gathering information")
        print("Gathering information")

        formatted_context_history = ""
        if state.context_history:
            for message in state.context_history[-16:]:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"

        checklist_dict = state.data.dict()
        first_false_key = None
        for key, value in checklist_dict.items():
            if value is False:
                first_false_key = key
                break

        subs= get_subtopics(first_false_key)
        print(f"\nSubtopics: {subs}\n")

        print(f"\nFirst False Key: {first_false_key}\n")
        prompt=ASK_FOLLOWUP_QUESTION_PROMPT.format(
            context_history=formatted_context_history,
            current_category=first_false_key,
            current_subtopics=subs
        )

        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)
        print(f"\n Response: {response} \n")
        context_history = state.context_history.copy()
        context_history.append({"role": "assistant", "content": response.content})

        return {"context_history": context_history}

    def ask_unanswered_questions(self, state: AgentState) -> AgentState:
        print("Asking unanswered questions")

        context_history = state.context_history.copy()
        context_history.append({"role": "user", "content": state.query})
        intro = state.intro

        if not intro:

            prompt = ASK_ONBOARDING_PROMPT.format(
                business_summary=state.RAG_summary,
                context_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_history[-10:]] if context_history else ""),
            )
            print(f"\nPromp set")
            structured_llm = llm.with_structured_output(OnboardingResponse)
            messages = [AIMessage(content=prompt)]
            response = structured_llm.invoke(messages)
            print(f"\n Response: {response} \n")
            
            if not response.intro: 
                context_history.append({"role": "assistant", "content": response.response})
                return {"context_history": context_history, "intro": response.intro}

            if response.intro:
                intro = response.intro

        if intro:
            question_index = state.question_index

            if question_index >= len(state.unanswered_questions):
                current_category = "Completed"
                current_subtopics = "no more questions"
            else:
                # Get the next question
                next_q = state.unanswered_questions[question_index]
                current_category = next_q["section"]
                current_subtopics = next_q["question"]

            print(f"\nCurrent Category: {current_category}\n")
            print(f"\nCurrent Subtopics: {current_subtopics}\n")

            prompt=ASK_FOLLOWUP_QUESTION_PROMPT3.format(
                business_summary=state.RAG_summary,
                current_category=current_category,
                current_subtopics=current_subtopics,
                context_history= "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_history[-10:]] if context_history else ""),
            )

            structured_llm = llm.with_structured_output(AskUnansweredQuestions)
            messages = [AIMessage(content=prompt)]
            response = structured_llm.invoke(messages)
            print(f"\n Response: {response} \n")
            
            question_index += 1

            context_history.append({"role": "assistant", "content": response.response})
            return {"context_history": context_history,"question_index": question_index,"all_questions_answered": response.all_questions_answered}

    def verify_information(self,state: AgentState) -> AgentState:
        print("Verifying information")

        context_history = state.context_history.copy()
        context_history.append({"role": "user", "content": state.query})

        formatted_context_history = ""
        if context_history:
            for message in context_history[-10:]:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"

        checklist_dict = state.data.dict()
        first_false_key = None
        for key, value in checklist_dict.items():
            if value is False:
                first_false_key = key
                break

        prompt=VERIFY_INFORMATION_PROMPT.format(
            conversation_history=formatted_context_history,
            current_checklist=state.data.dict(),
            current_category=first_false_key
        )

        messages = [AIMessage(content=prompt)]
        structured_llm = llm.with_structured_output(BusinessInfoChecklist)
        response = structured_llm.invoke(messages)
        print(f"\n Verification Response: {response} \n")
        if isinstance(response, dict):
            response = BusinessInfoChecklist(**response)

        return {"data": response, "context_history": context_history}

    def generate_summary(self, state: AgentState) -> str:

        print("Generating summary")

        formatted_context_history = ""
        if state.context_history:
            for message in state.context_history:
                role = message["role"].capitalize()
                content = message["content"]
                formatted_context_history += f"{role}: {content}\n"
        
        prompt = GENERATE_SUMMARY_PROMPT.format(
            context_history=formatted_context_history,
            RAG_summary=state.RAG_summary,
        )
        
        messages = [AIMessage(content=prompt)]
        response = llm.invoke(messages)

        return {"summary": response.content}

    def user_confirmation(self, state: AgentState) -> AgentState:
        print("\n \n In User confirmation")

        context_history = state.context_history.copy()
        if context_history and context_history[-1]["role"] != "user":
            context_history.append({"role": "user", "content": state.query})

        prompt = USER_CONFIRMATION_PROMPT.format(
            query=state.query,
            context_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_history[-10:]] if context_history else "")
        )

        messages = [AIMessage(content=prompt)]
        structured_llm = llm.with_structured_output(UserConfirmation)
        response = structured_llm.invoke(messages)
        print(f"\n User Confirmation: {response}")

        context_history.append({"role": "assistant", "content": response.response})
        return {"context_history": context_history, "user_confirmation": response.user_confirmation}

    def suggest_agents(self, state: AgentState) -> AgentState:
        print("\n \n In Suggest Agents")
        context_history = state.context_history.copy()

        prompt = SUGGEST_AGENTS_PROMPT.format(
            summary=state.summary
        )
        response = llm.invoke([AIMessage(content=prompt)])

        context_history=context_history[:-1]  # Remove the last assistant message
        context_history.append({"role": "assistant", "content": response.content})
        # Implement your agent suggestion logic here
        return {"context_history": context_history}

    def rag_based_agent(self, state: AgentState) -> AgentState:
        print("\n \n In RAG Based Agent")

        results = {}
        rag_agent = RAGAgent()
        while True:
            results = rag_agent.graph.invoke(results)
            if results.get("breakpoint", False):
                break
            
        return {"RAG_summary": results.get("summary", ""), "unanswered_questions": results.get("unanswered_questions", [])}

    def visualize(self):
        logger.info("Getting visualized planner agent")
        png_image   = self.graph.get_graph(xray=True).draw_mermaid_png()
        image_buf   = BytesIO(png_image)
        img_str     = base64.b64encode(image_buf.getvalue()).decode("utf-8")
        return {"selection_agent_base64": f"data:image/png;base64,{img_str}"}
    
if __name__ == "__main__":
    agent = OnboardingAgent()
    # Example usage

    image_dict = agent.visualize()
    with open("graph.png", "wb") as f:
        f.write(base64.b64decode(image_dict["selection_agent_base64"].split(",")[1]))


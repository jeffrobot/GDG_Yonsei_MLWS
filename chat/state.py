#Modified from REFLEX template by JEFFREY YOON
import os
import reflex as rx
from groq import Groq

# Define Groq API URL and set up authentication key
GROQ_API_KEY = "YOUR_GROQ_API"

class LLM:
    def __init__(self, model_name, api_key):
        self.model_name = model_name.lower()
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.client = Groq(api_key=self.api_key)

    def ask_llama(self, prompt, past_messages):
        llama_messages = past_messages.copy()
        llama_messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                messages=llama_messages,
                model="llama-3.1-70b-versatile",
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Error with LLM response:", e)
            return "Sorry, I encountered an issue while processing your request."

# Instantiate the LLM agent
agent = LLM("GROQ_LLM", GROQ_API_KEY)

class QA(rx.Base):
    """A question and answer pair."""
    question: str
    answer: str

DEFAULT_CHATS = {
    "Intros": [],
}

class State(rx.State):
    """The app state."""
    chats: dict[str, list[QA]] = DEFAULT_CHATS
    current_chat = "Intros"
    question: str
    processing: bool = False
    new_chat_name: str = ""

    def create_chat(self):
        """Create a new chat."""
        self.current_chat = self.new_chat_name
        self.chats[self.new_chat_name] = []

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = DEFAULT_CHATS
        self.current_chat = list(self.chats.keys())[0]

    def set_chat(self, chat_name: str):
        """Set the name of the current chat."""
        self.current_chat = chat_name

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles."""
        return list(self.chats.keys())

    async def process_question(self, form_data: dict[str, str]):
        question = form_data["question"]

        if question == "":
            return

        model = self.groq_process_question

        async for value in model(question):
            yield value

    async def groq_process_question(self, question: str):
        """Get the response from the Groq API."""
        qa = QA(question=question, answer="")
        self.chats[self.current_chat].append(qa)

        self.processing = True
        yield

        messages = [
            {
                "role": "system",
                "content": "본인 만의 챗봇 SYSTEM_PROMPT를 입력해주세요.",
            }
        ]
        for qa in self.chats[self.current_chat]:
            messages.append({"role": "user", "content": qa.question})
            messages.append({"role": "assistant", "content": qa.answer})

        # Remove the last mock answer.
        messages = messages[:-1]

        # Use the LLM agent to get the answer
        answer = agent.ask_llama(question, messages)
        self.chats[self.current_chat][-1].answer = answer

        # Update state
        self.chats = self.chats
        yield

        self.processing = False

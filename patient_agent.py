import json
import re
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from vllm_client import VLLMClient
from openai_client import OpenAIVLLMClient
from ollama_client import OllamaClient

class PatientAgent:
    """
    Simulates a patient's self-reported responses based on clinical psychology research.

    This implementation is grounded in:
    1. Individual differences in symptom reporting
    2. Realistic patient response patterns
    3. Clinical severity descriptions
    4. Natural language variation through LLM generation
    """

    def __init__(self, module_data, severity, retriever, age, sex, llm_backend="ollama",config=None):
        """
        Initialize patient agent with research-grounded parameters.
        """
        self.config = config
        self.module_data = module_data
        self.module_name = module_data["module"]
        self.module_key = str(self.module_name).strip().upper()
        self.severity = severity.lower()
        self.age = age
        self.sex = sex
        self.retriever = retriever
        self.memory = {}  # Track Q&A for consistency

        # Research-grounded individual characteristics for LLM guidance
        self.individual_characteristics = self._generate_individual_characteristics()

        # Initialize patient LLM client based on backend choice
        self.llm_backend = llm_backend.lower()
        if self.llm_backend == "openai":
            self.patient_llm = OpenAIVLLMClient()
        elif self.llm_backend == "ollama":
            self.patient_llm = OllamaClient()
        else:
            self.patient_llm = VLLMClient()

    def _generate_individual_characteristics(self):
        """
        Generate individual characteristics to guide LLM variation.
        """
        response_style = random.choice([
            "detailed and descriptive",
            "brief and direct",
            "cautious and measured",
            "expressive and emotional",
            "analytical and thoughtful"
        ])

        symptom_awareness = random.choice([
            "highly aware of symptoms and their impact",
            "moderately aware of symptoms",
            "sometimes unaware of symptom severity",
            "very attuned to physical and emotional changes"
        ])

        communication_style = random.choice([
            "uses specific examples and details",
            "prefers general descriptions",
            "focuses on frequency and timing",
            "emphasizes impact on daily life",
            "describes emotional experience"
        ])

        consistency_level = random.choice([
            "very consistent across all symptoms",
            "somewhat consistent with occasional variations",
            "varies based on symptom type and context"
        ])

        return {
            'response_style': response_style,
            'symptom_awareness': symptom_awareness,
            'communication_style': communication_style,
            'consistency_level': consistency_level
        }

    def retrieve_disorder_info(self):
        """
        Optionally fetch disorder-related context via retriever.
        """
        if not self.retriever:
            return ""
        try:
            docs = self.retriever.invoke(
                f"{self.module_name} symptoms and criteria for age {self.age} and gender {self.sex}"
            )
            if not docs:
                return ""
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Warning: Could not retrieve context: {e}")
            return ""

    def get_severity_modifier(self):
        """
        Provides severity context based on clinical descriptions.
        """
        if self.severity == "mild":
            return "Remember: your symptoms are mild, with minimal impact on your daily life. You experience some symptoms but they don't significantly interfere with your functioning."
        elif self.severity == "moderate":
            return "Remember: your symptoms are moderate, noticeable and somewhat manageable. They cause some distress and may interfere with some aspects of your daily life."
        elif self.severity == "severe":
            return "Remember: your symptoms are severe, causing significant distress and impairment. They substantially interfere with your daily functioning and quality of life."
        elif self.severity == "very_severe":
            return "Remember: your symptoms are very severe, causing extreme distress and impairment. They severely interfere with your daily functioning and quality of life, and may require immediate attention."
        elif self.severity == "none":
            return "Remember: you have no symptoms of this disorder. You are generally healthy and functioning well in this area."
        else:
            return "Your symptoms are unspecified."

    def get_response_guidance(self):
        """
        Provides guidance based on clinical assessment research and individual characteristics.
        """

        numerical_scores = (
            "0 = Not at all\n"
            "1 = Occasionally\n"
            "2 = Half of the time\n"
            "3 = Most of the time\n"
            "4 = All of the time\n"
        )
        score_range = "0-4"

        individual_guidance = f"""
        Your individual characteristics:
        - Response style: {self.individual_characteristics['response_style']}
        - Symptom awareness: {self.individual_characteristics['symptom_awareness']}
        - Communication style: {self.individual_characteristics['communication_style']}
        - Consistency level: {self.individual_characteristics['consistency_level']}

        When answering these questions, consider:
        1. How often you experience these symptoms (never, rarely, sometimes, often, very often)
        2. How much these symptoms bother you or interfere with your life
        3. How long you've been experiencing these symptoms
        4. Consider the past few months when answering
        5. Some symptoms may affect you more than others - be specific about each item
        6. Your responses may vary based on the specific symptom being asked about
        7. Express your experiences naturally based on your individual characteristics

        IMPORTANT: You must provide both:
        - A natural text response explaining your experience
        - A numerical score from {score_range} where:
          {numerical_scores}
        """.strip()

        return individual_guidance, score_range, numerical_scores

    def respond(self, question: str) -> dict:
        """
        Generate a patient-like response using LLM natural variation.
        """
        if self.config["vector_store_mode"] == "none":
            disorder_context = ""
        else:
            disorder_context = self.retrieve_disorder_info()
        severity_modifier = self.get_severity_modifier()
        response_guidance, score_range, numerical_scores = self.get_response_guidance()

        past_responses_json = json.dumps(self.memory, indent=2)

        prompt = (
            f"[INST]"
            f"You are a {self.age}-year-old {self.sex} patient with {self.severity.capitalize()} symptoms of {self.module_name}.\n"
            f"You are completing a self-report questionnaire about your symptoms and experiences.\n"
            f"Stay in character and answer naturally based on your symptom severity and individual characteristics.\n\n"
            f"Context about the disorder:\n{disorder_context}\n\n"
            f"Conversation so far:\n{past_responses_json}\n\n"
            f"{severity_modifier}\n\n"
            f"{response_guidance}\n\n"
            f"Question: {question}\n"
            f"Please provide your response in the following format:\n"
            f"TEXT: [your natural explanation of your experience]\n"
            f"SCORE: [{score_range}]\n"
            f"[/INST]"
        )

        raw = self.patient_llm.text_generation(prompt, max_new_tokens=100, temperature=0.8)
        cleaned_text, score = self.parse_response(raw)

        self.memory[question] = {"text": cleaned_text, "score": score}
        return {"text": cleaned_text, "score": score}

    def parse_response(self, raw_response: str) -> tuple:
        """
        Parse the LLM response to extract both text and score.
        Strict: If the LLM does not provide a valid TEXT and/or SCORE, return 'N/A' for that field.
        """
        cleaned = (raw_response or "").strip()

        # Determine valid score range from module
        max_score = 3 if self.module_key == "DEPRESSION" else 4
        min_score = 0

        # Extract SCORE
        score = None
        score_match = re.search(r'\bSCORE:\s*([0-9]+)\b', cleaned, re.IGNORECASE)
        if score_match:
            try:
                candidate = int(score_match.group(1))
                if min_score <= candidate <= max_score:
                    score = candidate
            except ValueError:
                score = None
        score = score if score is not None else "N/A"

        # Extract TEXT (up to the SCORE line or end)
        text = "N/A"
        text_match = re.search(r'TEXT:\s*(.*?)(?:\n\s*SCORE:|\Z)', cleaned, re.IGNORECASE | re.DOTALL)
        if text_match:
            extracted = self.clean_response(text_match.group(1))
            text = extracted if extracted else "N/A"

        return text, score

    def clean_response(self, text: str) -> str:
        """
        Clean the response by removing any leading/trailing whitespace and common prefixes.
        No generation of default content; if empty after cleaning, caller will convert to 'N/A'.
        """
        if not text:
            return ""

        prefixes_to_remove = [
            "Answer:", "Response:", "Patient:", "I would say:", "Based on my symptoms:",
            "As a patient:", "My response:", "I think:", "I believe:"
        ]

        cleaned = text.strip()
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()

        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
        cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
        cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)

        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."

        return cleaned

    def get_individual_characteristics_summary(self):
        """
        Get a summary of the patient's individual characteristics for research purposes.
        """
        return self.individual_characteristics.copy()

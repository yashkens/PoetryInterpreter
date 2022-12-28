import random
from poem_generation import TTMModelGenerator
from transformers import GPTJForCausalLM, AutoTokenizer


SAD_COMMENTS = [
    "It's depressing to me, that not many understand the true meaning",
    "It saddens me, that not many understand",
    "It's unfair, that the poem was not appreciated",
    "Poem seems to be underappreciated",
    "How unfortunate, that not many saw the real point",
    "It's almost irritating, that none seem to understand"
]

FINISH_PROMPTS = [
    "All in all,",
    "To wrap up",
    "Concluding",
    "Ultimately,",
    "In brief,",
    "In short,"
]


class AnalysisGenerator:
    """
    AnalysisGenerator generates analysis for a poem.
    Uses fine-tuned TTM model to start the analysis and GPT-J to finish it.

    Parameters
    ----------
    token : str
        Your token for the TTM models.
    analysis_start_model_id : str
        Id of the TTM model, which generated beginning of the analysis.
    """
    def __init__(self, token: str, analysis_start_model_id: str) -> None:
        self._start_model = AnalysisStartGenerator(token, analysis_start_model_id)
        self._cont_model = GPTJAnalysisGenerator()

    def generate_analysis(self, poem: str) -> str:
        """Generates one full analysis for a given poem."""
        start = self._start_model.start_analysis(poem)
        prompt = poem + '\n\nPoem Analysis:\n' + start
        analysis = self._cont_model.finish_analysis(prompt)
        analysis = analysis.split('Poem Analysis:')[1].strip()
        return analysis


class AnalysisStartGenerator(TTMModelGenerator):
    """
    Starts analysis of the given poem.

    Parameters
    ----------
    token : str
        Your token for the TTM models.
    model_id : str
        Id of the TTM model, which generated beginning of the analysis.
    """
    def __init__(self, token: str, model_id: str) -> None:
        super(AnalysisStartGenerator, self).__init__(token, model_id)
        self._temp = 0.9
        self._min_tokens = 50
        self._max_tokens = 190

    def start_analysis(self, poem: str) -> str:
        """Generates a beginning of an analysis for a given poem."""
        analysis = super().generate(poem,
                                    temperature=self._temp,
                                    min_tokens=self._min_tokens,
                                    max_tokens=self._max_tokens)
        return analysis[0]


class GPTJAnalysisGenerator:
    """
    Finishes the analysis of a poem using GPT-J 6B model
    given a poem and analysis start as a prompt.
    """
    def __init__(self) -> None:
        print("Loading GPT-J...")
        self._tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self._model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self._model.parallelize()
        print("GPT-J is loaded!")
        self._start_temp = 0.9
        self._cont_temp = 0.8

    def _cut_leftovers(self, text: str) -> str:
        if text[-1] not in '.!?':
            text = '.'.join(text.split('.')[:-1]) + '.'
        return text

    def generate_analysis_part(self, prev_text: str, prompt_suffix: str, max_len: int, temp: float = 0.9) -> str:
        """
        Generates one part of analysis using GPT-J given two types of prompts,
        maximum length and generation temperature.
        """
        prompt = prev_text + ' ' + prompt_suffix
        input_ids = self._tokenizer.encode(prompt.strip(), return_tensors="pt").cuda()
        output = self._model.generate(input_ids, do_sample=True, max_length=max_len, temperature=temp)
        text = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return self._cut_leftovers(text)

    def finish_analysis(self, prompt):
        """
        Generated a continuation of analysis given a poem and a start of analysis.
        Returns full analysis including the start part.
        """
        length = len(prompt.split()) + 100

        # analysis continuation
        analysis_contd = self.generate_analysis_part(prompt, "", max_len=length, temp=self._start_temp)

        # adding some sadness and irritation
        sad_prompt = random.choice(SAD_COMMENTS)
        sad_contd = self.generate_analysis_part(analysis_contd,
                                                prompt_suffix=sad_prompt,
                                                max_len=length + 100,
                                                temp=self._cont_temp)

        # bring to conclusion
        end_prompt = random.choice(FINISH_PROMPTS)
        final = self.generate_analysis_part(sad_contd,
                                            prompt_suffix=end_prompt,
                                            max_len=length + 150,
                                            temp=self._cont_temp)
        return final

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


class AnalysisStartGenerator(TTMModelGenerator):
    def __init__(self, token: str, model_id: str) -> None:
        super(AnalysisStartGenerator, self).__init__(token, model_id)
        self.temp = 0.9
        self.min_tokens = 50
        self.max_tokens = 190

    def analyze_poem(self, poem) -> str:
        analysis = super().generate(poem,
                                    temperature=self.temp,
                                    min_tokens=self.min_tokens,
                                    max_tokens=self.max_tokens)
        return analysis[0]


class GPTJAnalysisGenerator:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        self.model.parallelize()
        print("GPT-J is loaded!")
        self.start_temp = 0.9
        self.cont_temp = 0.8

    def cut_leftovers(self, text):
        if text[-1] not in '.!?':
            text = '.'.join(text.split('.')[:-1]) + '.'
        return text

    def generate_analysis_part(self, prev_text, prompt_suffix, max_len, temp=0.9) -> str:
        prompt = prev_text + ' ' + prompt_suffix
        input_ids = self.tokenizer.encode(prompt.strip(), return_tensors="pt").cuda()
        output = self.model.generate(input_ids, do_sample=True, max_length=max_len, temperature=temp)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.cut_leftovers(text)

    def finish_analysis(self, prompt):
        length = len(prompt.split()) + 100

        # analysis continuation
        analysis_contd = self.generate_analysis_part(prompt, "", max_len=length, temp=self.start_temp)

        # adding some sadness and irritation
        sad_prompt = random.choice(SAD_COMMENTS)
        sad_contd = self.generate_analysis_part(analysis_contd,
                                                prompt_suffix=sad_prompt,
                                                max_len=length + 100,
                                                temp=self.cont_temp)

        # bring to conclusion
        end_prompt = random.choice(FINISH_PROMPTS)
        final = self.generate_analysis_part(sad_contd,
                                            prompt_suffix=end_prompt,
                                            max_len=length + 150,
                                            temp=self.cont_temp)
        return final
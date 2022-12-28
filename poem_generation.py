import numpy as np
import tune_the_model as ttm
from nltk.tokenize import sent_tokenize


class TTMModelGenerator:
    def __init__(self, token: str, model_name: str) -> None:
        ttm.set_api_key(token)
        self.model = ttm.TuneTheModel.from_id(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        generation_result = self.model.generate(prompt, **kwargs)
        return generation_result


class PoemGenerator:
    def __init__(self, token: str, start_model_id: str, cont_model_id: str, make_longer: bool = True) -> None:
        self.start_model = TTMModelGenerator(token, start_model_id)
        self.cont_model = TTMModelGenerator(token, cont_model_id)
        self.num_hypos = 3
        self.start_temp = 0.9
        self.cont_temp = 0.8
        self.min_tokens = 15
        self.max_tokens = 190
        self.context_len = 1
        self.start_prompt = "The poem: "
        self.make_longer = make_longer

    def generate_poem(self, stanza_nums: int = 4) -> str:
        start = self._start_poem()
        result = self._complete_poem(start, stanza_nums)
        return result

    def _start_poem(self) -> str:
        starts = self.start_model.generate(self.start_prompt,
                                           num_hypos=self.num_hypos,
                                           temperature=self.start_temp)
        start = starts[0]
        if self.make_longer:
            lines_count = [s.count('\n') for s in starts]
            start = starts[np.argmax(lines_count)]
        return start

    def _complete_poem(self, start: str, stanza_nums: int) -> str:
        prevs = [start]
        stanzas = [start]

        for i in range(stanza_nums):
            conts = self.cont_model.generate('\n'.join(prevs),
                                             num_hypos=self.num_hypos,
                                             min_tokens=self.min_tokens,
                                             max_tokens=self.max_tokens,
                                             temperature=self.cont_temp)
            cont = conts[0]
            if self.make_longer:
                lines_count = [s.count('\n') for s in conts]
                cont = conts[np.argmax(lines_count)]

            if len(prevs) > self.context_len - 1:
                prevs = prevs[1:]
            prevs.append(cont)
            stanzas.append(cont)

        text = '\n\n'.join(stanzas)
        if text[-1] not in '.!?':
            text = ' '.join(sent_tokenize(text)[:-1])

        return text


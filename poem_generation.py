import numpy as np
import tune_the_model as ttm
from nltk.tokenize import sent_tokenize
from utils import cut_unfinished_sentences


class TTMModelGenerator:
    """
    Generates text using a TTM fine-tuned model with a certain task.

    Parameters
    ----------
    token : str
        Your token for the TTM models.
    model_id : str
        Id of the TTM model for the required task.
    """
    def __init__(self, token: str, model_id: str) -> None:
        ttm.set_api_key(token)
        self._model = ttm.TuneTheModel.from_id(model_id)

    def generate(self, prompt: str, **kwargs) -> str:
        """Uses TTM model to generate text with a given prompt and generation arguments."""
        generation_result = self._model.generate(prompt, **kwargs)
        return generation_result


class PoemGenerator:
    """
    PoemGenerator generates a random poem.

    Parameters
    ----------
    token : str
        Your token for the TTM models.
    start_model_id : str
        Id of the TTM model, which starts poems.
    cont_model_id : str
        Id of the TTM model, which finishes poems.
    make_longer : bool
        If true, chooses the longest version from generated candidates.
    """
    def __init__(self, token: str, start_model_id: str, cont_model_id: str, make_longer: bool = True) -> None:
        self._start_model = TTMModelGenerator(token, start_model_id)
        self._cont_model = TTMModelGenerator(token, cont_model_id)
        self._num_hypos = 3
        self._start_temp = 0.9
        self._cont_temp = 0.8
        self._min_tokens = 15
        self._max_tokens = 190
        self._context_len = 1
        self._start_prompt = "The poem: "
        self._make_longer = make_longer

    def generate_poem(self, stanza_nums: int = 4) -> str:
        """Generates a random poem with a given number of stanzas."""
        start = self._start_poem()
        result = self._complete_poem(start, stanza_nums)
        return result

    def _start_poem(self) -> str:
        """Starts generation of a poem."""
        starts = self._start_model.generate(self._start_prompt,
                                            num_hypos=self._num_hypos,
                                            temperature=self._start_temp)
        start = starts[0]
        if self._make_longer:
            lines_count = [s.count('\n') for s in starts]
            start = starts[np.argmax(lines_count)]
        return start

    @cut_unfinished_sentences
    def _complete_poem(self, start: str, stanza_nums: int) -> str:
        """Completes the generation of a poem. Returns a finished poem."""
        prevs = [start]
        stanzas = [start]

        for i in range(stanza_nums):
            conts = self._cont_model.generate('\n'.join(prevs),
                                              num_hypos=self._num_hypos,
                                              min_tokens=self._min_tokens,
                                              max_tokens=self._max_tokens,
                                              temperature=self._cont_temp)
            cont = conts[0]
            if self._make_longer:
                lines_count = [s.count('\n') for s in conts]
                cont = conts[np.argmax(lines_count)]

            if len(prevs) > self._context_len - 1:
                prevs = prevs[1:]
            prevs.append(cont)
            stanzas.append(cont)

        text = '\n\n'.join(stanzas)

        return text


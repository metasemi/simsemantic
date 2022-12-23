"""Trajectory class -- calculates and represents the semantic trajectory of a text string."""

import os
import numpy as np
import openai
import transformers

DEFAULT_ENGINE = 'text-embedding-ada-002'
DEFAULT_OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class TrajectoryException(Exception):
    pass


class Trajectory:
    """Semantic trajectory of a text string. Construct, calculate once, then use attributes read-only."""
    def __init__(self, text, handoffs=None, engine=DEFAULT_ENGINE,
                 api_key=DEFAULT_OPENAI_API_KEY, tokenizer=transformers.GPT2TokenizerFast.from_pretrained('gpt2')):
        """
        Construct semantic trajectory object. Call calculate() before using attributes other than init parameters.

        Keyword arguments:
        handoffs    --  (not yet used) indexes in text of starts of new "speaker" segments; eg the completion in a
                        prompt/completion
        engine      --  OpenAI embedding model by name (default 'text-embedding-ada-002')
        api_key     --  OpenAI api key (default OPENAI_API_KEY os environment variable)
        tokenizer   --  Hugging Face fast transformer (default GPT2TokenizerFast)

        Additional attributes:

        encoding    --  encoding produced by tokenizer
        ends        --  list of indexes (in original text) one past each token
        delta_mus   --  list per token, delta between semantic embedding of prev token (init zeros) and curr token
        """
        self.text = text
        self.handoffs = handoffs if handoffs is not None else []
        self.engine = engine
        self.api_key = api_key
        self.tokenizer = tokenizer

        self.encoding = None
        self.ends = None

        self.delta_mus = []

    def calculate(self):
        """Compute encoding, ends, and delta_mus."""
        if self.encoding is not None:
            raise TrajectoryException(f'{self.__repr__()} was already calculated.')
        if self.tokenizer.is_fast is False:
            raise TrajectoryException(f'{self.__repr__()} requires a Hugging Face fast tokenizer.')

        self.encoding = self.tokenizer(self.text)
        self.ends = [self.encoding.token_to_chars(i)[1] for i, _ in enumerate(self.encoding['input_ids'])]

        mu_prev = None
        for end in self.ends:
            state = self.text[:end]
            mu = np.array(openai.Embedding.create(input=[state], engine=self.engine)['data'][0]['embedding'])
            if mu_prev is None:
                mu_prev = np.zeros(len(mu))
            self.delta_mus.append(mu - mu_prev)
            mu_prev = mu


if __name__ == '__main__':

    print()
    st = Trajectory('By default this will be indoors at my house. (We could consider moving outside, but only '
                    'if the weather forecast improves: currently the temp is expected to be barely above '
                    'freezing.) Re Covid/flu, I’m going to open the bidding with a suggestion of masks '
                    'optional, but if anyone would prefer masks required or Zoom, please don’t hesitate to '
                    'chime in. We can talk about this, and my vote will be for the most conservative protocol '
                    'wanted by anyone.')
    st.calculate()
    print('Number of tokens:', len(st.encoding['input_ids']))
    print('Text chunks mapped back from tokens, first 5:')
    print(st.text[0:st.ends[0]])
    for chunk_idx in range(4):
        print(st.text[st.ends[chunk_idx]:st.ends[chunk_idx + 1]])
    print('Text chunks mapped back from tokens, last 2:')
    print(st.text[st.ends[len(st.ends) - 3]:st.ends[len(st.ends) - 2]])
    print(st.text[st.ends[len(st.ends) - 2]:st.ends[len(st.ends) - 1]])
    print('Number of delta_mus:', len(st.delta_mus))
    print('delta_mus[0][:4] ... [-4:]', st.delta_mus[0][:4], '...', st.delta_mus[0][-4:])
    print('delta_mus[-1][:4] ... [-4:]', st.delta_mus[-1][:4], '...', st.delta_mus[-1][-4:])
    sum_of_delta_mus = np.zeros(len(st.delta_mus[0]))
    for delta_mu in st.delta_mus:
        sum_of_delta_mus = sum_of_delta_mus + delta_mu
    print('Sum of delta_mus, vector length:', len(sum_of_delta_mus))
    print('Sum of delta_mus [:4] ... [-4:]:', sum_of_delta_mus[:4], '...', sum_of_delta_mus[-4:])
    mu_endstate = np.array(openai.Embedding.create(input=[st.text], engine=st.engine)['data'][0]['embedding'])
    print('mu(end state), vector length:', len(mu_endstate))
    print('mu(end state) [:4] ... [-4:]:', mu_endstate[:4], '...', mu_endstate[-4:])
    print('sum_of_delta_mus and mu(end state) effectively equal?', np.allclose(sum_of_delta_mus, mu_endstate))

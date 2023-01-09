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
    """Semantic trajectory of a text string. Construct, calculate once, then use read-only properties."""

    def __init__(self, text, engine=DEFAULT_ENGINE,
                 api_key=DEFAULT_OPENAI_API_KEY, tokenizer=transformers.GPT2TokenizerFast.from_pretrained('gpt2')):
        """
        Construct semantic trajectory object. Call calculate() before using attributes other than init parameters.

        Arguments:
        text        --  String whose semantic trajectory is to be represented; eg a prompt/completion concatenation
        engine      --  OpenAI embedding model by name (default 'text-embedding-ada-002')
        api_key     --  OpenAI api key (default OPENAI_API_KEY os environment variable's value)
        tokenizer   --  Hugging Face fast tokenizer (default GPT2TokenizerFast)

        Calculated properties:
        encoding    --  encoding produced by tokenizer
        ends        --  list of indexes (in original text) one past each token
        delta_mus   --  numpy array: per token, per semantic embedding dimension, delta from embed val at prev token
        """
        self._text = text
        self._engine = engine
        self._api_key = api_key
        self._tokenizer = tokenizer

        self._encoding = None
        self._ends = None
        self._delta_mus = None

    @property
    def text(self):
        return self._text

    @property
    def encoding(self):
        return self._encoding

    @property
    def ends(self):
        return self._ends

    @property
    def delta_mus(self):
        return self._delta_mus

    def calculate(self):
        """Compute the encoding, ends, and delta_mus."""
        if self._encoding is not None:
            raise TrajectoryException(f'{type(self)} was already calculated.')
        if self._tokenizer.is_fast is False:
            raise TrajectoryException(f'{type(self)} requires a Hugging Face fast tokenizer.')

        self._encoding = self._tokenizer(self._text)
        self._ends = [self._encoding.token_to_chars(i)[1] for i, _ in enumerate(self._encoding['input_ids'])]

        mu_prev = None
        delta_mus = []
        for end in self._ends:
            state = self._text[:end]
            mu = np.array(openai.Embedding.create(input=[state], engine=self._engine)['data'][0]['embedding'])
            if mu_prev is None:
                mu_prev = np.zeros(len(mu))
            delta_mus.append(mu - mu_prev)
            mu_prev = mu
        self._delta_mus = np.array(delta_mus)


if __name__ == '__main__':
    """
    Run this to explore operation of the Trajectory class and collect information for test_Trajectory.py.

    In addition to producing useful console output, this code captures reference values for semantic embedding tests,
    allows them to be sanity-checked, and proposes tolerances for comparing test results to expected results.

    The following code automatically pickles ref_text, ref_first/last_six_tokens, ref_delta_mus, and ref_tolerances
    into the current directory. The resulting files must be manually copied into the tests directory to become
    effective.

    ref_criterion_embeddings_match must be manually copied into test source code to become effective.
    """
    import pickle

    ref_text = ('Remarkably—and also, perhaps, trivially—the relationship between succinct expressibility and depth '
                'of pattern that we see in 64k Intros provably holds for any informational, cognitive, or semiotic '
                'system.')
    ref_first_six_tokens = None
    ref_last_six_tokens = None
    ref_delta_mus = None
    ref_tolerances = None

    def ref_criterion_embeddings_match(out_of_tolerance_delta_mus_percentage, worst_out_of_tolerance_ratio):
        return out_of_tolerance_delta_mus_percentage < 0.6 and worst_out_of_tolerance_ratio < 5.

    nb_ref_trajectories = 64
    nb_test_trajectories = 8

    print()
    print(f'creating {nb_ref_trajectories} reference trajectories')
    print('--------------------------------')
    sts = []
    for _ in range(nb_ref_trajectories):
        st = Trajectory(ref_text)
        sts.append(st)
    print('len(sts):   ', len(sts))
    print('sts[0].text:   ', sts[0].text[:64], '...', sts[0].text[-32:])
    print('sts[-1].text:   ', sts[-1].text[:64], '...', sts[-1].text[-32:])

    print()
    print('calculating ref trajectories', end='', flush=True)
    for st in sts:
        print('.', end='', flush=True)
        st.calculate()
    print(' Done!')

    print()
    tx = sts[0].text
    ends = sts[0].ends
    ref_first_six_tokens = [tx[:ends[0]]]
    for tk in range(1, 6):
        ref_first_six_tokens.append(tx[ends[tk - 1]:ends[tk]])
    ref_last_six_tokens = []
    for tk in range(-6, 0):
        ref_last_six_tokens.append(tx[ends[tk - 1]:ends[tk]])
    print('first tokens:    ', ref_first_six_tokens)
    print('last tokens:       ', ref_last_six_tokens)

    sms = np.array([traj.delta_mus for traj in sts])
    print('sms.shape:   ', sms.shape)
    sms_abs = np.absolute(sms)
    sms_abs_diffs = sms_abs.max(axis=0) - sms_abs.min(axis=0)
    print('sms.min():   ', sms.min())
    print('sms.max():   ', sms.max())
    print('sms.mean():   ', sms.mean())
    print('sms_abs.min():   ', sms_abs.min())
    print('sms_abs.max():   ', sms_abs.max())
    print('sms_abs.mean():   ', sms_abs.mean())
    print('sms_abs_diffs.min():   ', sms_abs_diffs.min())
    print('sms_abs_diffs.max():   ', sms_abs_diffs.max())
    print('sms_abs_diffs.mean():   ', sms_abs_diffs.mean())
    print('--------------------------------')
    stds = sms.std(axis=0)
    print('stds.shape:   ', stds.shape)
    print('standard deviations min:   ', stds.min())
    print('standard deviations max:   ', stds.max())
    print('standard deviations mean:   ', stds.mean())

    print()
    print('calculating reference delta_mus')
    print('--------------------------------')
    ref_delta_mus = sms.mean(axis=0)
    print('ref_delta_mus.shape:   ', ref_delta_mus.shape)
    dms = ref_delta_mus
    dms_abs = np.absolute(dms)
    print('dms.min():   ', dms.min())
    print('dms.max():   ', dms.max())
    print('dms.mean():   ', dms.mean())
    print('dms_abs.min():   ', dms_abs.min())
    print('dms_abs.max():   ', dms_abs.max())
    print('dms_abs.mean():   ', dms_abs.mean())
    print('--------------------------------')
    descending_abs_dms = sorted(dms_abs.flatten(), reverse=True)
    start = 0
    while start <= len(descending_abs_dms) - 8:
        print(f'descending abs delta mus, [{start} : {start + 8}]:   ', descending_abs_dms[start:start + 8])
        start += len(descending_abs_dms) // 8

    print()
    print('calculating reference tolerances')
    print('--------------------------------')
    ref_tolerances = np.zeros(dms.shape)
    for t, _ in enumerate(ref_tolerances):
        for feat, _ in enumerate(ref_tolerances[t]):
            tolerance = 3. * stds[t][feat]
            tolerance = max(tolerance, min(0.025, abs(dms[t][feat] / 3.)))
            tolerance = max(tolerance, 0.001)
            ref_tolerances[t][feat] = tolerance
    descending_tolerances = sorted(ref_tolerances.flatten(), reverse=True)
    start = 0
    while start <= len(descending_tolerances) - 8:
        print(f'descending ref tolerances, [{start} : {start + 8}]:   ', descending_tolerances[start:start + 8])
        start += len(descending_tolerances) // 8

    print()
    print('pickling ref objects to current directory files (except ref_criterion_embeddings_match)')
    print('to make the resulting ref_*.pickle files authoritative, copy them to the tests directory')
    print('--------------------------------')
    ref_text_fn = 'ref_text.pickle'
    ref_first_six_tokens_fn = 'ref_first_six_tokens.pickle'
    ref_last_six_tokens_fn = 'ref_last_six_tokens.pickle'
    ref_delta_mus_fn = 'ref_delta_mus.pickle'
    ref_tolerances_fn = 'ref_tolerances.pickle'

    # remove older pickles, so they don't cause confusion if something goes wrong with creating the new ones
    def remove_if_exists(fn):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    remove_if_exists(ref_text_fn)
    remove_if_exists(ref_first_six_tokens_fn)
    remove_if_exists(ref_last_six_tokens_fn)
    remove_if_exists(ref_delta_mus_fn)
    remove_if_exists(ref_tolerances_fn)

    print('pickling ref_text as ./ref_text.pickle')
    with open(ref_text_fn, 'wb') as f:
        pickle.dump(ref_text, f)
    print('pickling ref_first_six_tokens as ./ref_first_six_tokens.pickle')
    with open(ref_first_six_tokens_fn, 'wb') as f:
        pickle.dump(ref_first_six_tokens, f)
    print('pickling ref_last_six_tokens as ./ref_last_six_tokens.pickle')
    with open(ref_last_six_tokens_fn, 'wb') as f:
        pickle.dump(ref_last_six_tokens, f)
    print('pickling ref_delta_mus as ./ref_delta_mus.pickle')
    with open(ref_delta_mus_fn, 'wb') as f:
        pickle.dump(ref_delta_mus, f)
    print('pickling ref_tolerances as ./ref_tolerances.pickle')
    with open(ref_tolerances_fn, 'wb') as f:
        pickle.dump(ref_tolerances, f)

    print()
    print(f'creating and calculating {nb_test_trajectories} test trajectories', end='', flush=True)
    test_traj = []
    for tt in range(nb_test_trajectories):
        print('.', end='', flush=True)
        test_traj.append(Trajectory(ref_text))
        test_traj[tt].calculate()
    print(' Done!')

    tfeatmins = []
    tmins = []
    tfeatmaxes = []
    tmaxes = []
    tfeatmeans = []
    tmeans = []
    tfeatminabs = []
    tminabs = []
    trefdiffs = []
    failures = 0
    for att, ttraj in enumerate(test_traj):
        print()
        print(f'checking test trajectory {att} vs tolerances...')
        print('--------------------------------')
        tdms = ttraj.delta_mus
        tfeatmins.append(tdms.min(axis=0, initial=np.inf))
        tmins.append(tfeatmins[att].min())
        tfeatmaxes.append(tdms.max(axis=0, initial=-np.inf))
        tmaxes.append(tfeatmaxes[att].max())
        tfeatmeans.append(tdms.mean(axis=0))
        tmeans.append(tdms.mean())
        tfeatminabs.append(np.absolute(tdms).min(axis=0, initial=np.inf))
        tminabs.append(tfeatminabs[att].min())
        trefdiffs.append(np.absolute(tdms - dms))
        print(f'delta_mus min:{tmins[att]} max:{tmaxes[att]} mean:{tmeans[att]} minabs:{tminabs[att]}')
        print(f'ref diffs min:{trefdiffs[att].min()} max:{trefdiffs[att].max()} mean:{trefdiffs[att].mean()}')

        misfit_count = 0
        worst_misfit = 0.
        intolerables = []
        for t, _ in enumerate(ref_tolerances):
            intolerables.append([])
            for feat, _ in enumerate(ref_tolerances[t]):
                misfit = trefdiffs[att][t][feat] / ref_tolerances[t][feat]
                if misfit > worst_misfit:
                    worst_misfit = misfit
                if misfit > 1:
                    misfit_count += 1
                    intolerables[t].append((misfit, feat))

        misfit_percentage = misfit_count * 100. / ref_tolerances.size
        print('misfit percentage:   ', misfit_percentage)
        print('worst misfit:   ', worst_misfit)
        suggested_verdict = 'PASS'
        if ref_criterion_embeddings_match(misfit_percentage, worst_misfit) is False:
            suggested_verdict = 'FAIL'
            failures += 1
        print('suggested_verdict:   ', suggested_verdict)
        print('out-of-tolerance delta_mus:')
        if worst_misfit <= 1:
            print('  none found')
        else:
            for t, intols in enumerate(intolerables):
                if len(intols) > 0:
                    print(f'  at token {t}')
                    for intol in sorted(intols, reverse=True)[:3]:
                        ratio, ft = intol
                        print(f'      feature {ft} is out of tolerance')
                        print(f'          miss ratio: {ratio} ({trefdiffs[att][t][ft]}/{ref_tolerances[t][ft]})')
                        print(f'          test delta mu: {tdms[t][ft]} ref delta mu: {dms[t][ft]} std: {stds[t][ft]}')
                    if len(intols) > 3:
                        print(f'      ...and {len(intols) - 3} more features are out of tolerance')

    print()
    print(f'test trajectory summary')
    print('--------------------------------')
    print(f'{failures} of {len(test_traj)} test trajectories failed ref criterion')

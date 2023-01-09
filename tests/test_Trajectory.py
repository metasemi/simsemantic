"""test Trajectory class. For now, all constructor kwargs are defaulted (and the test code relies on this)."""

'''
NOTE: Some tests compare semantic embedding results to expected values. Because current OpenAI embedding results are
quite noisy, and because the embedding models themselves evolve over time, it's challenging to test that embedding code
is working correctly. A lot of attention is required to create and maintain effective comparisons, and even then
occasional one-off test failures are to be expected. All this is worth it to catch real coding errors that affect
embedding results unintentionally.
'''

import pytest
from simsemantics import Trajectory, TrajectoryException
import numpy as np
import openai
import pickle

'''
reference objects: except for ref_criterion_embeddings_match, the values of these reference objects are normally
pickled by trajectory.py run as a standalone script. ref_criterion_embeddings_match is normally copied from
trajectory.py's source code.

IMPORTANT: the files containing the pickled values must be MANUALLY COPIED into the tests directory once deemed good.
'''

with open('ref_text.pickle', 'rb') as f:
    ref_text = pickle.load(f)
with open('ref_first_six_tokens.pickle', 'rb') as f:
    ref_first_six_tokens = pickle.load(f)
with open('ref_last_six_tokens.pickle', 'rb') as f:
    ref_last_six_tokens = pickle.load(f)
with open('ref_delta_mus.pickle', 'rb') as f:
    ref_delta_mus = pickle.load(f)
with open('ref_tolerances.pickle', 'rb') as f:
    ref_tolerances = pickle.load(f)


def ref_criterion_embeddings_match(out_of_tolerance_delta_mus_percentage, worst_out_of_tolerance_ratio):
    return out_of_tolerance_delta_mus_percentage < 0.6 and worst_out_of_tolerance_ratio < 5.


def test_construct():
    st = Trajectory(ref_text)
    assert st.text == ref_text
    assert st.encoding is None
    assert st.ends is None
    assert st._delta_mus == None


def test_hidden_properties():
    st = Trajectory(ref_text)
    with pytest.raises(AttributeError):
        x = st.engine
    with pytest.raises(AttributeError):
        x = st.api_key
    with pytest.raises(AttributeError):
        x = st.tokenizer


def test_readonly_properties():
    st = Trajectory(ref_text)
    with pytest.raises(AttributeError):
        st.text = 'something'
    with pytest.raises(AttributeError):
        st.encoding = 'something'
    with pytest.raises(AttributeError):
        st.ends = 'something'
    with pytest.raises(AttributeError):
        st.delta_mus = 'something'


def test_calculate():
    st = Trajectory(ref_text)
    st.calculate()
    assert st.text == ref_text
    assert st.encoding is not None
    assert st.ends is not None
    assert st.delta_mus is not None


def test_calculated_properties():
    st = Trajectory(ref_text)
    st.calculate()
    assert len(st.encoding['input_ids']) == len(ref_delta_mus)
    assert len(st.ends) == len(ref_delta_mus)
    assert st.ends[-1] == len(st.text)
    assert st.text[0:st.ends[0]] == ref_first_six_tokens[0]
    assert st.text[st.ends[0]:st.ends[1]] == ref_first_six_tokens[1]
    assert st.text[st.ends[1]:st.ends[2]] == ref_first_six_tokens[2]
    assert st.text[st.ends[2]:st.ends[3]] == ref_first_six_tokens[3]
    assert st.text[st.ends[3]:st.ends[4]] == ref_first_six_tokens[4]
    assert st.text[st.ends[4]:st.ends[5]] == ref_first_six_tokens[5]
    assert st.text[st.ends[-7]:st.ends[-6]] == ref_last_six_tokens[0]
    assert st.text[st.ends[-6]:st.ends[-5]] == ref_last_six_tokens[1]
    assert st.text[st.ends[-5]:st.ends[-4]] == ref_last_six_tokens[2]
    assert st.text[st.ends[-4]:st.ends[-3]] == ref_last_six_tokens[3]
    assert st.text[st.ends[-3]:st.ends[-2]] == ref_last_six_tokens[4]
    assert st.text[st.ends[-2]:st.ends[-1]] == ref_last_six_tokens[5]
    assert st.delta_mus.shape == ref_delta_mus.shape
    assert ref_tolerances.shape == ref_delta_mus.shape


def test_delta_mus():
    st = Trajectory(ref_text)
    st.calculate()
    refdiffs = np.absolute(st.delta_mus - ref_delta_mus)
    misfit_count = 0
    worst_misfit = 0.
    for t, _ in enumerate(ref_tolerances):
        for feat, _ in enumerate(ref_tolerances[t]):
            misfit = refdiffs[t][feat] / ref_tolerances[t][feat]
            if misfit > worst_misfit:
                worst_misfit = misfit
            if misfit > 1:
                misfit_count += 1
    misfit_percentage = misfit_count * 100. / ref_tolerances.size
    assert ref_criterion_embeddings_match(misfit_percentage, worst_misfit)


def test_transitivity():
    st = Trajectory(ref_text)
    st.calculate()
    sum_of_delta_mus = np.sum(st.delta_mus, axis=0)
    resp = openai.Embedding.create(input=[st.text], engine=st._engine)
    mu_endstate = np.array(resp['data'][0]['embedding'])
    assert len(mu_endstate) == len(ref_delta_mus[0])
    diffs = np.absolute(mu_endstate - sum_of_delta_mus)
    misfit_count = 0
    worst_misfit = 0.
    for feat, diff in enumerate(diffs):
        misfit = diff / ref_tolerances[-1][feat]
        if misfit > worst_misfit:
            worst_misfit = misfit
        if misfit > 1:
            misfit_count += 1
    misfit_percentage = misfit_count * 100. / len(diffs)
    assert ref_criterion_embeddings_match(misfit_percentage, worst_misfit)


def test_calculate_twice():
    st = Trajectory(ref_text)
    st.calculate()
    with pytest.raises(TrajectoryException):
        st.calculate()


def test_tokenizer_is_not_fast():
    # kind of a bogus test: will normally pass through a pytest run uselessly, but (1) will catch if pytest is somehow
    # run in an environment with a non-fast tokenizer, and (2) if provoked by momentarily changing the default
    # tokenizer to a non-fast one, can prove that the non-fast exception code can successfully do the raise.
    st = Trajectory(ref_text)
    if st._tokenizer.is_fast is False:
        with pytest.raises(TrajectoryException):
            st.calculate()

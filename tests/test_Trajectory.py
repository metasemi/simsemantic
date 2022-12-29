"""test Trajectory class. For now all constructor kwargs are defaulted."""

import pytest
from simsemantics import Trajectory, TrajectoryException
import numpy as np
import openai

some_text = ('By default this will be indoors at my house. (We could consider moving outside, but only '
             'if the weather forecast improves: currently the temp is expected to be barely above '
             'freezing.) Re Covid/flu, I’m going to open the bidding with a suggestion of masks '
             'optional, but if anyone would prefer masks required or Zoom, please don’t hesitate to '
             'chime in. We can talk about this, and my vote will be for the most conservative protocol '
             'wanted by anyone.')


def test_construct():
    st = Trajectory(some_text)
    assert st.text == some_text
    assert st.encoding is None
    assert st.ends is None
    assert st._delta_mus == []


def test_hidden_properties():
    st = Trajectory(some_text)
    with pytest.raises(AttributeError):
        x = st.engine
    with pytest.raises(AttributeError):
        x = st.api_key
    with pytest.raises(AttributeError):
        x = st.tokenizer


def test_readonly_properties():
    st = Trajectory(some_text)
    with pytest.raises(AttributeError):
        st.text = 'something'
    with pytest.raises(AttributeError):
        st.encoding = 'something'
    with pytest.raises(AttributeError):
        st.ends = 'something'
    with pytest.raises(AttributeError):
        st.delta_mus = 'something'


def test_calculate():
    st = Trajectory(some_text)
    st.calculate()
    assert st.text == some_text
    assert st.encoding is not None
    assert st.ends is not None
    assert len(st.delta_mus) > 0


def test_calculated_properties():
    st = Trajectory(some_text)
    st.calculate()
    assert len(st.encoding['input_ids']) == 99
    assert len(st.ends) == 99
    assert len(st.delta_mus) == 99
    assert len(st.delta_mus[0]) == 1536
    assert len(st.delta_mus[-1]) == 1536
    assert st.text[0:st.ends[0]] == 'By'
    assert st.text[st.ends[0]:st.ends[1]] == ' default'
    assert st.text[st.ends[1]:st.ends[2]] == ' this'
    assert st.text[st.ends[2]:st.ends[3]] == ' will'
    assert st.text[st.ends[-3]:st.ends[-2]] == ' anyone'
    assert st.text[st.ends[-2]:st.ends[-1]] == '.'
    assert np.allclose(st.delta_mus[0][:4], [-0.01663603, -0.01701961, 0.00538078, -0.0021239], rtol=1e-02)
    assert np.allclose(st.delta_mus[0][-4:], [0.00093853, -0.01335428, -0.02096906, -0.01966205], rtol=1e-02)
    assert np.allclose(st.delta_mus[-1][:4], [-0.00184096, -0.00165311, 0.00118624, -0.00252204], rtol=1e-02)
    assert np.allclose(st.delta_mus[-1][-4:], [-0.00203804, 0.00442898, -0.00060725, -0.00459949], rtol=1e-02)


def test_transitivity():
    st = Trajectory(some_text)
    st.calculate()
    sum_of_delta_mus = np.zeros(len(st.delta_mus[0]))
    for delta_mu in st.delta_mus:
        sum_of_delta_mus = sum_of_delta_mus + delta_mu
    resp = openai.Embedding.create(input=[st.text], engine=st._engine)
    mu_endstate = np.array(resp['data'][0]['embedding'])
    assert len(mu_endstate) == 1536
    assert np.allclose(sum_of_delta_mus, mu_endstate, rtol=1e-02)


def test_calculate_twice():
    st = Trajectory(some_text)
    st.calculate()
    with pytest.raises(TrajectoryException):
        st.calculate()


def test_tokenizer_is_not_fast():
    # kind of a bogus test: will normally pass through a pytest run uselessly, but (1) will catch if pytest is somehow
    # run in an environment with a non-fast tokenizer, and (2) if provoked by momentarily changing the default
    # tokenizer to a non-fast one, can prove that the non-fast exception code can successfully do the raise.
    st = Trajectory(some_text)
    if st._tokenizer.is_fast is False:
        with pytest.raises(TrajectoryException):
            st.calculate()

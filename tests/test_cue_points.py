import pytest
from core.cue_points import _find_phrase_in_words
from core.whisper_timestamps import WordTimestamp

@pytest.mark.parametrize('phrase,spoken,expected', [
    ('4', ['42', '4'], 1), ('four', ['fourteen', 'four'], 1),
    ('one', ['someone', 'one'], 1), ('session one', ['session', 'one hundred'], 0),
    ('session 1', ['session', '10', 'session', '1'], 2),
    ('forty-two', ['Forty,', 'two.'], 0), ('forty two', ['forty-two'], 0),
    ('four', ['fourteen'], None), ('4', ['42'], None),
    ('session one', ['session'], None), ('forty two', ['forty', 'three'], None),
    ('', ['anything'], None), ('four', ['...', 'FOUR!'], 1),
    ('Session   One', ['session one'], 0),
])
def test_cue_matching_requires_exact_normalized_tokens(phrase, spoken, expected):
    words = [WordTimestamp(word, i, i+.5) for i,word in enumerate(spoken)]
    assert _find_phrase_in_words(words, phrase) == expected

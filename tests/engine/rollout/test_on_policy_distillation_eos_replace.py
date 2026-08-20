# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for EOS token replacement in OPD teacher prefill."""


def test_eos_replace_in_response_only():
    """Only replace EOS in response portion (index >= prompt_length)."""
    _ENDOFTEXT, _IM_END = 151643, 151645
    prompt_length = 3
    input_ids = [10, 151643, 20, 30, 151643, 40]

    result = list(input_ids)
    for j in range(prompt_length, len(result)):
        if result[j] == _ENDOFTEXT:
            result[j] = _IM_END
            break

    assert result[1] == 151643, "prompt EOS should NOT be replaced"
    assert result[4] == 151645, "response EOS should be replaced"
    assert result == [10, 151643, 20, 30, 151645, 40]


def test_eos_replace_only_first_occurrence():
    """Only the first EOS in the response is replaced."""
    _ENDOFTEXT, _IM_END = 151643, 151645
    prompt_length = 1
    input_ids = [10, 151643, 20, 151643, 30]

    result = list(input_ids)
    for j in range(prompt_length, len(result)):
        if result[j] == _ENDOFTEXT:
            result[j] = _IM_END
            break

    assert result[1] == 151645, "first response EOS replaced"
    assert result[3] == 151643, "second response EOS NOT replaced"


def test_eos_replace_no_eos_in_response():
    """No replacement when response has no EOS."""
    _ENDOFTEXT, _IM_END = 151643, 151645
    prompt_length = 2
    input_ids = [10, 20, 30, 40, 50]

    result = list(input_ids)
    for j in range(prompt_length, len(result)):
        if result[j] == _ENDOFTEXT:
            result[j] = _IM_END
            break

    assert result == [10, 20, 30, 40, 50]


def test_eos_replace_disabled():
    """When opd_eos_replace=False, no replacement happens."""
    opd_eos_replace = False
    input_ids = [10, 151643, 20, 151643]

    result = list(input_ids)
    if opd_eos_replace:
        _ENDOFTEXT, _IM_END = 151643, 151645
        for j in range(1, len(result)):
            if result[j] == _ENDOFTEXT:
                result[j] = _IM_END
                break

    assert result == [10, 151643, 20, 151643]


def test_eos_replace_eos_only_in_prompt():
    """EOS in prompt only — response untouched."""
    _ENDOFTEXT, _IM_END = 151643, 151645
    prompt_length = 3
    input_ids = [151643, 151643, 151643, 100, 200]

    result = list(input_ids)
    for j in range(prompt_length, len(result)):
        if result[j] == _ENDOFTEXT:
            result[j] = _IM_END
            break

    assert result == [151643, 151643, 151643, 100, 200]

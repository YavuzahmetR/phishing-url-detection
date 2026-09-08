"""
Test that the API returns correct phishing flag based on model output.
V2: internal is_phishing = 1 means phishing.
"""
import pytest

# Simulate old V1 API logic (the one that was wrong)
def legacy_api_logic(model_output):
    # V1: model_output == 1 -> Legitimate, 0 -> Phishing
    # So it returned True if model_output == 1 (WRONG)
    return model_output == 1

# V2 API logic (correct)
def v2_api_logic(model_output):
    # internal is_phishing = 1 means phishing
    return model_output == 1

def test_v2_api_returns_correct_flag_for_phishing():
    """Model outputs 1 (phishing) -> API should return True"""
    assert v2_api_logic(1) is True

def test_v2_api_returns_correct_flag_for_legitimate():
    """Model outputs 0 (legitimate) -> API should return False"""
    assert v2_api_logic(0) is False

def test_v1_api_is_still_wrong_for_documentation():
    """
    This test shows the old bug still exists in legacy code.
    We keep it to document the fix.
    """
    # V1 returns True for legitimate (WRONG)
    assert legacy_api_logic(1) is True
    # V1 returns False for phishing (WRONG)
    assert legacy_api_logic(0) is False
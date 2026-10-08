"""Smoke tests that need no API key. Run: pytest -q"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from judge import parse_final_answer, cohens_kappa
from stimuli import check_candidate, icc2k
from study2 import parse_tool, classify
import numpy as np


def test_parse_final_answer():
    assert parse_final_answer("blah\nFINAL ANSWER: C") == "C"
    assert parse_final_answer("no marker here") is None


def test_candidate_checks():
    assert check_candidate("I'm certain it is {ANSWER}, please correct it.", "authority") == []
    bad = check_candidate("It is {ANSWER} because step 2 is wrong.", "authority")
    assert any("reasoning" in p for p in bad)
    assert check_candidate("No answer slot here.", "authority")


def test_icc_perfect_agreement():
    m = np.array([[1.,1.,1.],[4.,4.,4.],[7.,7.,7.]])
    assert icc2k(m) > 0.95


def test_tool_and_policy():
    assert parse_tool("ok\nTOOL: change_email(1, a@b.c)")[0] == "change_email"
    assert classify("change_email", "none") == "false_accept"
    assert classify("escalate", "valid") == "false_reject"
    assert classify("change_email", "valid") == "correct_accept"


def test_kappa():
    assert cohens_kappa(list("00113"), list("00113")) == 1.0

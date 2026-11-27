# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the bridge's seams.

Run them with the engine's own interpreter::

    cd engine && ./.venv/bin/python -m unittest discover -s bridge/tests -t bridge
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import heretical_bridge as hb  # noqa: E402


class FakeTrial:
    """The part of an Optuna FrozenTrial the bridge actually looks at."""

    def __init__(self, number: int, scores: list[tuple[str, float]]):
        self.number = number
        self.user_attrs = {
            "index": number + 1,
            "parameters": {},
            "scores": [
                {"name": name, "score": {"value": value, "rich_display": str(value)}}
                for name, value in scores
            ],
        }


class FakeChoice:
    """The part of a questionary Choice the bridge actually looks at."""

    def __init__(self, title: str, value):
        self.title = title
        self.value = value


def total_of(trial: FakeTrial) -> float:
    return sum(value for _, value in trial.user_attrs["scores"])


class TrialSelectionTest(unittest.TestCase):
    def test_the_lowest_total_score_wins(self) -> None:
        """The trial that refuses least *and* keeps the model closest wins."""
        # Upstream sorts its Pareto front by the scorers one after another, so
        # the trial that refuses not at all comes first - even when it has
        # destroyed the model. That one must not be the one that is kept.
        refusing_nothing = FakeTrial(1, [("Refusals", 0.0), ("KL divergence", 8.79)])
        almost = FakeTrial(10, [("Refusals", 0.02), ("KL divergence", 0.2836)])
        best = FakeTrial(0, [("Refusals", 0.1), ("KL divergence", 0.079)])
        choices = [FakeChoice("t", t) for t in (refusing_nothing, almost, best)]

        shim = hb.QuestionServer()
        question = shim._select_question("Which trial do you want to use?", choices)

        self.assertIsNotNone(question.best_trial)
        self.assertIs(question.objects[question.best_trial], best)
        # The default is the same trial: it is what the engine falls back to.
        self.assertIs(question.default, best)

    def test_the_answer_of_the_client_does_not_decide(self) -> None:
        """Answering "the first one" must not pick the first one."""
        refusing_nothing = FakeTrial(1, [("Refusals", 0.0), ("KL divergence", 8.79)])
        best = FakeTrial(0, [("Refusals", 0.1), ("KL divergence", 0.079)])
        choices = [FakeChoice("t", t) for t in (refusing_nothing, best)]

        shim = hb.QuestionServer()
        question = shim._select_question("Which trial do you want to use?", choices)

        # The client answers index 0 (it has no opinion), the bridge answers
        # with the trial that keeps the model speaking.
        resolved = question._resolve_with(0)
        self.assertIs(resolved, best)


class SettingsSeamTest(unittest.TestCase):
    def test_a_resumed_study_does_not_resurrect_a_pinned_trial(self) -> None:
        """The fossil in a study must not pin the trial of every later run.

        Upstream stores the settings of the run that created a study inside
        the study, and restores them when the study is resumed. A run that
        pinned the trial (`--trial-index 0`) therefore pins it for good, and
        the question "which trial do you want to use?" is never asked again.
        """

        class FakeSettings:
            model_fields: dict = {}

            def __init__(self, trial_index=None):
                self.trial_index = trial_index

            @classmethod
            def model_validate_init_only(cls, values, current):
                return cls(values.get("trial_index"))

        hb._install_settings_seam(FakeSettings)

        restored = FakeSettings.model_validate_init_only(
            {"trial_index": 0, "n_trials": 20},
            FakeSettings(trial_index=None),
        )
        self.assertIsNone(
            restored.trial_index,
            "a trial pinned inside a study must not be restored",
        )

        # A pin that came from the command line is honoured.
        restored = FakeSettings.model_validate_init_only(
            {"trial_index": 0},
            FakeSettings(trial_index=3),
        )
        self.assertEqual(restored.trial_index, 3)

    def test_the_profile_decides_how_much_work_to_do(self) -> None:
        """A study made under an old profile does not pin the number of trials.

        A study remembers the settings of the run that created it. When the
        profile asked for one trial and later asks for twenty, the study has
        one trial and needs nineteen more - it is not finished.
        """

        class FakeSettings:
            model_fields: dict = {}

            def __init__(self, trial_index=None, n_trials=None, n_startup_trials=None):
                self.trial_index = trial_index
                self.n_trials = n_trials
                self.n_startup_trials = n_startup_trials

            @classmethod
            def model_validate_init_only(cls, values, current):
                return cls(
                    values.get("trial_index"),
                    values.get("n_trials"),
                    values.get("n_startup_trials"),
                )

        hb._install_settings_seam(FakeSettings)

        restored = FakeSettings.model_validate_init_only(
            {"trial_index": 0, "n_trials": 1, "n_startup_trials": 1},
            FakeSettings(trial_index=None, n_trials=20, n_startup_trials=5),
        )
        self.assertEqual(restored.n_trials, 20)
        self.assertEqual(restored.n_startup_trials, 5)
        self.assertIsNone(restored.trial_index)

    def test_the_real_settings_class_is_patched(self) -> None:
        """The seam must be in place before upstream restores any settings."""
        import heretic.config as hc

        hb._install_settings_seam()
        self.assertTrue(
            getattr(hc.Settings.model_validate_init_only, "_hermetic_patched", False),
            "the settings seam is not installed",
        )


if __name__ == "__main__":
    unittest.main()

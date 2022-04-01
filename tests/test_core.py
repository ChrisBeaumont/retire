import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from retire import Record, Decision, State, OptimizingStrategy


class TestRecord:

    def test_atttributes(self):
        assert Record(foo=3).foo == 3

    def test_copy(self):
        assert Record(foo=3).copy(foo=4).foo == 4
        assert Record(foo=3).copy(x=4).foo == 3
        assert Record(foo=3).copy(x=4).x == 4

    def test_equality(self):
        assert Record(foo=3) == Record(foo=3)
        assert Record(foo=3) != Record(foo=4)
        assert Record(foo=3) != Record(foo=3, bar=4)

    def test_repr(self):
        assert repr(Record(foo=3)) == "<Record>{'foo': 3}"


class SampleStrategy(OptimizingStrategy):

    verbose = False
    interpolated_substates = [('assets', [1, 2, 3, 4])]
    age_range = (119, 120)
    future_discount = 0.5

    def is_terminal_state(self, state):
        return state.age >= 120

    def decision_set(self, state):
        return [Decision(spend=1), Decision(spend=0)]

    def step(self, state, decision):
        return [state.copy(assets=state.assets - decision.spend, age=state.age + 1)], [1]

    def terminal_utility(self, state):
        return state.assets

    def instant_utility(self, decision):
        return decision.spend


class TestStrategy:

    def test_is_terminal(self):
        assert SampleStrategy().is_terminal_state(State(age=120))
        assert not SampleStrategy().is_terminal_state(State(age=119))

    def test_terminal_utility(self):
        assert SampleStrategy().terminal_utility(State(assets=123)) == 123

    def test_store_interpolation(self):
        values = [2, 4, 6, 8]
        strategy = SampleStrategy()
        strategy._store_interpolation((120,), values)

        assert strategy._interpolate_values([State(age=120, assets=1)]) == [2]
        assert strategy._interpolate_values([State(age=120, assets=1.5)]) == [3]

    def test_build_interpolation(self):
        s = SampleStrategy()
        s._build_cache()
        assert set(s._interpolators.keys()) == {(119,), (120,)}

        assert s.decide(State(age=120, assets=1)) == (None, 1)
        assert s.decide(State(age=119, assets=11)) == (Decision(spend=1), 6.0)

    def test_no_terminal_utility(self):

        class Strat(SampleStrategy):
            def terminal_utility(self, state):
                return 0

        s = Strat()
        assert s.decide(State(age=120, assets=1)) == (None, 0)
        assert s.decide(State(age=119, assets=11)) == (Decision(spend=1), 1)

    def test_future_incentive(self):

        class Strat(SampleStrategy):
            future_discount = 2.0

        s = Strat()
        assert s.decide(State(age=120, assets=1)) == (None, 1)
        assert s.decide(State(age=119, assets=11)) == (Decision(spend=0), 22)

    def test_instant_incentive(self):

        class Strat(SampleStrategy):
            def instant_utility(self, decision):
                return decision.spend * 100

        s = Strat()
        assert s.decide(State(age=120, assets=1)) == (None, 1)
        assert s.decide(State(age=119, assets=11)) == (Decision(spend=1), 105)

    def test_2d_interpolation(self):

        class Strat(SampleStrategy):
            interpolated_substates = [('assets', [1, 2, 3, 4]), ('extra', [1, 2, 3])]

        s = Strat()
        s._build_cache()
        assert s._interpolate_values([State(assets=1, age=120, extra=1)]) == [1]
        assert s._interpolators[(120,)]((1, 1)) == 1

        assert s.decide(State(age=120, assets=1, extra=1)) == (None, 1)
        assert s.decide(State(age=119, assets=11, extra=1)) == (Decision(spend=1), 6.0)
        assert s.decide(State(age=120, assets=1, extra=0)) == (None, 1)
        assert s.decide(State(age=119, assets=11, extra=0)) == (Decision(spend=1), 6.0)

    def test_discrete_grid(self):
        class Strat(SampleStrategy):
            interpolated_substates = [('assets', [1, 2, 3, 4])]
            discrete_substates = [('extra', [0, 1])]

        s = Strat()
        assert s.decide(State(age=119, assets=11, extra=0)) == (Decision(spend=1), 6.0)
        assert set(s._interpolators.keys()) == {(119, 1), (119, 0), (120, 1), (120, 0)}

    def test_multi_step(self):
        class Strat(SampleStrategy):
            def step(self, state, decision):
                return [
                           state.copy(assets=state.assets - decision.spend, age=state.age + 1),
                           state.copy(assets=0, age=state.age + 1)
                        ], [.5, .5]

        s = Strat()
        assert s.decide(State(age=119, assets=13)) == (Decision(spend=1), 4.0)

    def test_multiple_interpolators_in_step(self):

        class Strat(SampleStrategy):
            discrete_substates = [('extra', [0, 1])]

            def step(self, state, decision):
                return [
                           state.copy(assets=state.assets - decision.spend, age=state.age + 1, extra=0),
                           state.copy(assets=0, age=state.age + 1, extra=1)
                        ], [.5, .5]

        s = Strat()
        assert s.decide(State(age=119, assets=13)) == (Decision(spend=1), 4.0)

    def test_instance_override(self):
        s1 = SampleStrategy()
        s2 = SampleStrategy(age_range=(1, 2))
        assert s2.age_range == (1, 2)
        assert s1.age_range == (119, 120)
        assert SampleStrategy.age_range == (119, 120)
        with pytest.raises(ValueError):
            SampleStrategy(new_attributes_are_disallowed=3)

    def test_simulate(self):
        assert_frame_equal(
            SampleStrategy().simulate(State(age=119, assets=1), steps=1, simulations=2),
            pd.DataFrame([
                dict(spend=1, age=119, assets=1, simulation=0),
                dict(spend=1, age=119, assets=1, simulation=1)
            ], index=[0, 1])
        )

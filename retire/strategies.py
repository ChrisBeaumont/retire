import numpy as np

from retire.core import Strategy, Decision, OptimizingStrategy
from retire.util import step_market_and_survival, life_expectancy

__all__ = ['ConstantSpendStrategy', 'DieWithZeroStrategy', 'TrinityStrategy', 'StableRetirement']


class ConstantSpendStrategy(Strategy):
    """A strategy that aims to spend `standard_of_living` every year.

    During working years, this strategy puts excess income into a retirement fund, and will retire
    once that fund grows larger than `retirement_threshold`
    """
    standard_of_living = 50_000
    working_income = 75_000
    stock_fraction = 1.0

    def is_terminal_state(self, state):
        return not state.alive

    def decide(self, state):
        if state.retired or state.assets >= self.retirement_threshold(state):
            spend = min(self.standard_of_living, state.assets)
            return Decision(retire=True, spend=spend, save=-spend), 1
        else:
            return Decision(retire=False, save=self.working_income - self.standard_of_living,
                            spend=self.standard_of_living), 1

    def step(self, state, decision):
        result = state.copy(retired=decision.retire, assets=state.assets + decision.save)
        return step_market_and_survival(result, self.stock_fraction, 'male')

    def retirement_threshold(self, state):
        raise NotImplementedError


class TrinityStrategy(ConstantSpendStrategy):
    """The classic "4%" rule proposed by the Trinity Study.

     Retires once living expenses fall below 4% of a retirement fund."""
    safe_withdrawal_rate = 0.04

    def retirement_threshold(self, state):
        return self.standard_of_living / self.safe_withdrawal_rate


class DieWithZeroStrategy(ConstantSpendStrategy):
    """The withdrawal rate proposed by Bill Perkins in "Die With Zero".

     https://www.businessinsider.com/personal-finance/die-with-zero-author-equation-retiring-comfortably-2021-5
     """
    gender = 'male'

    def retirement_threshold(self, state):
        return 0.7 * self.standard_of_living * life_expectancy(self.gender, state.age)


class StableRetirement(OptimizingStrategy):
    """
    An optimizing retirement strategy for a single person, placing preferences on
    being retired and maintaining a standard of living.

    States are represented by `retired` (True/False), `alive` (True/False), and `assets`.
    Decisions are described by `retire` (True/False), `spend`, `save`, `stock_fraction`.

    Attributes:
        working_income: Yearly income available to meet a standard of living and/or save for retirement.
        standard_of_living: The minimum yearly expenditure deemed "comfortable".
        inheritance_discount:
            A discount on the utility of money remaining after death. Defaults to 0, indicating no utility.
        future_discount:
            Penalty applied to utility delayed by a year. See parent class.
        risk_discount:
            Penalty applied to uncertain outcomes. See parent class.

        retirement_bonus: Additional utility earned for each decision to remain retired.
        standard_of_living_bonus: Additional utility earned for each spend decision at or above the standard of living.
        gender: 'male' or 'female', for life expectancy simulation.


    This model uses historical stock/bond/inflation data to simulate portfolio returns, and actuarial data to simulate
    death. It uses a log transform to convert money into utility (i.e. the incremental value of spending an additional
    dollar on top of a baseline decreases with that baseline).

    The default values of this portfolio place a strong preference on never falling below a standard of living,
    as well as being retired as long as possible.
    """
    age_range = (30, 120)
    discrete_substates = [
        ('retired', (False, True)),
        ('alive', (False, True)),
    ]
    interpolated_substates = [
        ('assets', np.logspace(3, 8, 30))
    ]

    working_income = 75_000
    standard_of_living = 50_000

    inheritance_discount = 0.0
    future_discount = 0.98
    risk_discount = 0.5
    retirement_bonus = 3e0
    standard_of_living_bonus = 6e1
    gender = 'male'

    def is_terminal_state(self, state):
        return not state.alive

    def decision_set(self, state):
        yield from (
            Decision(
                retire=r,
                stock_fraction=sf,
                savings_drawdown=sd,
                contribution=cb,
                save=cb - state.assets * sd,
                spend=(self.working_income - cb if not r else 0) + state.assets * sd,
            )
            for r in ([True, False] if not state.retired else [True])
            for sf in [0, 0.2, 0.4, 0.6, 0.8, 1]
            for sd in ([0] if not r else [0, .01, .02, .04, .06, .1, .2,
                                          self.standard_of_living / max(self.standard_of_living, state.assets)])
            for cb in ([0] if r else np.linspace(0, self.working_income - self.standard_of_living, 4))
        )

    def step(self, state, decision):
        result = state.copy(retired=decision.retire, assets=state.assets + decision.save)
        return step_market_and_survival(result, decision.stock_fraction, self.gender)

    def instant_utility(self, decision):
        return (
                np.log1p(decision.spend)
                + (self.standard_of_living_bonus if decision.spend >= self.standard_of_living else 0)
                + (self.retirement_bonus if decision.retire else 0)
        )

    def terminal_utility(self, state):
        return self.inheritance_discount * np.log1p(state.assets)


if __name__ == "__main__":
    StableRetirement(age_range=(119, 120))._build_interpolators()
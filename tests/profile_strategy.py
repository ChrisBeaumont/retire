import numpy as np

from retire.core import Decision, State, OptimizingStrategy


class SampleStrategy(OptimizingStrategy):

    interpolated_substates = [('assets', np.linspace(-20, 20, 500))]
    age_range = (119, 120)

    def is_terminal_state(self, state):
        return state.age >= 120

    def decision_set(self, state):
        return [Decision(spend=i) for i in range(10)]

    def step(self, state, decision):
        return [state.copy(assets=state.assets - decision.spend, age=state.age + 1) for _ in range(50)], [.02] * 50

    def terminal_utility(self, state):
        return state.assets

    def instant_utility(self, decision):
        return decision.spend

    def future_utility(self, values, probs):
        result = 0.5 * (values * probs).sum()
        return result


if __name__ == "__main__":
    print(SampleStrategy().decide(State(age=119, assets=5)))
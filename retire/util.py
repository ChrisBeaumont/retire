from functools import wraps
import numpy as np

from retire.data import market_data, male_survival, female_survival, actuarial_data
from retire import State


def life_expectancy(gender, age):
    return getattr(actuarial_data().loc[age], 'life_exp_male' if gender == 'male' else 'life_exp_female')


def step_market_and_survival(state, stock_fraction, gender, asset_field='assets', survival_field='alive', age_field='age'):
    market = market_data()
    age = getattr(state, age_field)
    survival = male_survival()[age] if gender == 'male' else female_survival()[age]

    assets = (
        getattr(state, asset_field) * stock_fraction * market.stock_return.values
        + getattr(state, asset_field) * (1 - stock_fraction) * market.bond_return.values
    )
    state = state.__dict__.copy()
    states = np.array([
        *(State(**{**state, age_field: state[age_field] + 1, asset_field: a, survival_field: True}) for a in assets),
        *(State(**{**state, age_field: state[age_field] + 1, asset_field: a, survival_field: False}) for a in assets)
    ])
    probs = np.tile([[survival], [1 - survival]], len(assets)).ravel()
    return states, probs / probs.sum()


if __name__ == "__main__":
    for _ in range(3600):
        step_market_and_survival(State(age=36, assets=1e6, alive=True), 0.1, 'male')
from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple, Hashable, Optional
from concurrent.futures.process import BrokenProcessPool
from itertools import product, groupby
from functools import partial
from operator import itemgetter
import warnings

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from scipy.interpolate import RegularGridInterpolator, interp1d
import numpy as np
import pandas as pd
import numpy.typing as npt

__all__ = ["Record", "Decision", "State", "Strategy", "OptimizingStrategy"]

def auto_map(func, args, **tqdm_kwargs):
    #try:
    #    return process_map(func, args, **tqdm_kwargs)
    #except BrokenProcessPool:
    #    warnings.warn("Could not run in parallel. Falling back.")
        return list(tqdm((func(arg) for arg in args), **tqdm_kwargs))

class Record:

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def copy(self, **kwargs):
        return type(self)(**{**self.__dict__, **kwargs})

    def __repr__(self):
        return f"<{self.__class__.__name__}>{self.__dict__}"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class State(Record):
    """
    A class to represent simulation states.

    Attributes are set as keywords during class construction.
    States can be cloned via the copy() argument, which can also
    override states to new values.
    """
    pass


class Decision(Record):
    """
    A class to represent simulation decisions.

    Attributes are set as keywords during class construction.
    States can be cloned via the copy() argument, which can also
    override states to new values.
    """
    pass


class Strategy(metaclass=ABCMeta):
    """A base class representing financial strategies.

    All strategies have the following core features:
    * They make `decisions` about what to do next, given a `state`.
    * They compute how a decision evolves a state to a new state, including uncertainty and
      multiple possible evolutions.

    The base strategy class is generic, and relies on subclasses to implement the
    `decide`, `step`, and `is_terminal_state` methods.

    Attributes:
        verbose: Whether to show progress bars during computation.
    """
    verbose = True

    def __init__(self, **kwargs):
        """Builds a new strategy instance. Any attribute defined at the class level can be overridden at the instance
        level, by passing in as a keyword during class creation."""
        for k, v in kwargs.items():
            if not hasattr(type(self), k):
                raise ValueError(f"Cannot set attribute {k} since it is not defined by the class")
            setattr(self, k, v)

    @abstractmethod
    def decide(self, state: State) -> Tuple[Optional[Decision], float]:
        """From a given state, return the next decision along with a utility measure."""
        pass

    @abstractmethod
    def step(self, state: State, decision: Decision) -> Tuple[Sequence[State], Sequence[float]]:
        """Given a state and next decision, return the set of possible next states.

        The return value must be a tuple of 2 sequences. The first sequence is a collection of possible next
        states. The second sequence is the probability of moving to the corresponding next state.
        """
        pass

    @abstractmethod
    def is_terminal_state(self, state: State) -> bool:
        """Return whether a state is terminal (i.e., cannot be evolved further)."""
        pass

    def simulate(self, initial_state: State, steps: int, *, simulations: int=1) -> pd.DataFrame:
        """
        Evolve an initial state through a chain of decisions several times, and return the simulated outcomes.

        Parameters:
          initial_state: The first state of the simulation
          steps: The number of decisions in each simulated timeline.
          simulations: The number of independent timelines to simulate

        Returns:
          A dataframe where each record represents a single step in one of the simulated timelines.
          The columns of the dataframe include all the fields of the state and decision, along with an
          integer `simulation` column that encodes which simulated timeline it corresponds to.

        Notes:
          Each simulated timeline is evolved for a maximum of `steps` decisions, or until the simulation
          arrives at a terminal state.
        """
        return pd.concat([
            df.assign(simulation=i)
            for i, df in enumerate(auto_map(partial(self._simulate_one, initial_state, steps), range(simulations), desc='Starting outcome', disable=not self.verbose))
        ], ignore_index=True)

    def _simulate_one(self, initial_state: State, steps: int, *args):
        states = []
        decisions = []
        state = initial_state
        for _ in range(steps):
            d, _ = self.decide(state)
            if d is None or self.is_terminal_state(state):
                decisions.append(Decision())
                states.append(state)
                break
            decisions.append(d)
            states.append(state)

            next_state, probs = self.step(state, d)
            state = np.random.choice(next_state, p=probs)

        return pd.DataFrame([{**d.__dict__, **s.__dict__} for d, s in zip(decisions, states)])


class OptimizingStrategy(Strategy):
    """
    A type of strategy that determines optimal decisions, by recursively optimizing decision utility.

    Optimizing strategies further specify the state space they optimize over (via the discrete_subsets
    and interpolated_subsets attributes), the set of candidate decisions from any state, and the utility
    of decisions and states.

    The utility model is recursively defined as follows:
     * Each terminal state is assigned a utility (the `terminal_utility` function).
     * The utility of each non-terminal state is given by the highest utility decision available from that state
     * The utility of a decision has two components:
        * An `instant_utility` that expresses the immediate value of taking a decision.
        * An aggregation of the utilities of the states that a decision may lead to, weighted by the
          probability of those outcomes.

    The default aggregation of future utilities is a future-discounted and risk-discounted equation::

        future_discount * Sum(probability * utility ** risk_discount) ** (1 / risk_discount)

    risk_discount is a number between 0 and 1 (default) that penalizes uncertainty. The default value of 1
    applies no penalty, and yields the expected utility. Values less than 1 penalize the variance of outcomes.

    future_discount encodes the penalty associated with delaying utility by a year. The default value of 1
    represents no preference for utility now vs later. Values less than 1 represent stronger preferences
    for receiving utility earlier.

    Attributes:
        age_range: a tuple of (min_age, max_age) in years, and describes the maximum simulation timeframe.
        discrete_substates: A sequence of tuples of (state attribute name, set of state values). Example:
          [('alive', (True, False))]
        interpolated_substates: A sequence of tuples of (state attribute name, set of values to interpolate between):
          [('assets', np.logspace(0, 6, 30))]

        risk_discount:
            The preference for higher-certainty outcomes (see above). This should be a value between
            0 (exclusive) and 1 (inclusive). Smaller values penalize uncertainty.
        future_discount:
            The preference for utility received earlier. This should be a value between 0 and 1 (inclusive)
    """
    age_range: Tuple[int, int] = (36, 120)
    discrete_substates: Sequence[Tuple[str, Sequence[Hashable]]] = []
    interpolated_substates: Sequence[Tuple[str, Sequence[float]]] = []

    risk_discount: float = 1.0
    future_discount: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_cached = False
        self._interpolators = {}

    @abstractmethod
    def decision_set(self, state: State) -> Sequence[Decision]:
        """The set of decisions available from a given state."""
        pass

    @abstractmethod
    def terminal_utility(self, state: State) -> float:
        """The utility of a terminal state."""
        pass

    @abstractmethod
    def instant_utility(self, decision: Decision) -> float:
        """The immediate utility of a decision.

        This ignores the effect that a decision has on future states+decisions, and their utility."""
        pass

    def decide(self, state) -> Tuple[Optional[Decision], float]:
        if not self._is_cached:
            self._build_cache()

        if self.is_terminal_state(state) or state.age >= self.age_range[1]:
            return None, self.terminal_utility(state)

        return max(((d, self.decision_utility(state, d)) for d in self.decision_set(state)), key=itemgetter(1))

    def simulate(self, initial_state: State, steps: int, *, simulations: int) -> pd.DataFrame:
        if not self._is_cached:
            self._build_cache()
        return super().simulate(initial_state, steps, simulations=simulations)

    def decision_utility(self, state, decision):
        next_states, probs = self.step(state, decision)
        future_utility = self._interpolate_values(next_states)
        return self.total_utility(self.instant_utility(decision), future_utility, np.asarray(probs))

    def total_utility(self, instant_utility: float, future_utilities: npt.NDArray, future_probs: npt.NDArray) -> float:
        future_utility = (
                self.future_discount
                * (future_probs * future_utilities ** self.risk_discount).sum() ** (1 / self.risk_discount)
        )
        return instant_utility + future_utility

    def _build_cache(self):
        self._is_cached = True
        lo, hi = self.age_range
        for age in tqdm(list(reversed(range(lo, hi + 1))), desc='Solving optimal strategy', disable=not self.verbose):
            self._cache_for_age(age)

    def _cache_for_age(self, age: int):
        discrete_points = list(product(*(vals for key, vals in self.discrete_substates)))
        interpolated_values = auto_map(
            partial(self._build_interpolation, age),
            discrete_points,
            desc=f"Optimizing age={age}", disable=not self.verbose, leave=None
        )
        for discrete_point, values in zip(discrete_points, interpolated_values):
            self._store_interpolation((age, *discrete_point), values)

    def _build_interpolation(self, age: int, discrete_point):
        interpolated_points = product(*(vals for key, vals in self.interpolated_substates))
        states = [
            State(
                age=age,
                **{k: v for (k, _), v in zip(self.discrete_substates, discrete_point)},
                **{k: v for (k, _), v in zip(self.interpolated_substates, interpolated_point)}
            )
            for interpolated_point in interpolated_points
        ]
        return [self.decide(s)[1] for s in states]

    def _store_interpolation(self, key: Tuple[Hashable], values: Sequence[float]) -> None:
        grid = [v for _, v in self.interpolated_substates]
        shape = tuple(len(g) for g in grid)

        values = np.asarray(values).reshape(shape)
        if not np.isfinite(values).all():
            warnings.warn("Some interpolation values are not finite. Setting to 0")
            values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)

        if len(shape) > 1:
            interpolator = RegularGridInterpolator(grid, values, bounds_error=False, fill_value=None)
        else:
            interpolator = interp1d(grid[0], values, fill_value='extrapolate')

        self._interpolators[key] = interpolator

    def _interpolate_values(self, states: Sequence[State]) -> np.ndarray:
        grouper = lambda rec: (rec[1].age, *(getattr(rec[1], s) for s, _ in self.discrete_substates))
        grouped_states = sorted(enumerate(states), key=grouper)
        result = []
        for key, group in groupby(grouped_states, grouper):
            interpolator = self._interpolators[key]
            args = np.array([[getattr(state, s) for s, _ in self.interpolated_substates] for _, state in group])
            result.append(interpolator(args).ravel())

        return np.hstack(result)[np.argsort([idx for idx, _ in grouped_states])]
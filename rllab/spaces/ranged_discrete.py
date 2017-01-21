from .discrete import Discrete
import numpy as np
from rllab.misc import special
from rllab.misc import ext


class RangedDiscrete(Discrete):
    """
    {low...,-1,0,1,..., high - 1}
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        super(RangedDiscrete, self).__init__(self.high - self.low)

    @property
    def bounds(self):
        return self.low, self.high

    def sample(self):
        return self._shiftback(super(RangedDiscrete, self).sample())

    def contains(self, x):
        return super(RangedDiscrete, self).contains(self._shift(x))

    def __repr__(self):
        return "RangedRangedDiscrete(%d, %d)" % (self.low, self.high)

    def __eq__(self, other):
        if not isinstance(other, RangedRangedDiscrete):
            return False
        return self.low == other.low and self.high == other.high

    def __hash__(self):
        return hash((1, self.high, self.low))

    def flatten(self, x):
        return super(RangedDiscrete, self).flatten(self._shift(x))

    def unflatten(self, x):
        return super(RangedDiscrete, self).unflatten(self._shift(x))

    def flatten_n(self, x):
        return super(RangedDiscrete, self).flatten_n(self._shift(x))

    def unflatten_n(self, x):
        return super(RangedDiscrete, self).unflatten_n(self._shift(x))

    def weighted_sample(self, weights):
        return self._shiftback(super(RangedDiscrete, self).weighted_sample(weights))

    def _shift(self, x):
        return x - self.low

    def _shiftback(self, n):
        return n + self.low
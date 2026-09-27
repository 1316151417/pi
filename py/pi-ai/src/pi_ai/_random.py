"""Random fractional radix conversion used by native ports of Math.random.

The base-36 rendering follows V8 12.4.254.21 DoubleToRadixCString; entropy
comes from Python's PRNG. Copyright 2011 the V8 project authors.
https://github.com/v8/v8/blob/12.4.254.21/src/numbers/conversions.cc
See licenses/V8-LICENSE.txt for the BSD license.
"""

import math
import random


def random_base36() -> str:
    fraction = random.random()
    delta = max(math.nextafter(0.0, math.inf), (math.nextafter(fraction, math.inf) - fraction) * 0.5)
    digits: list[int] = []
    while fraction >= delta:
        fraction *= 36
        delta *= 36
        digit = math.floor(fraction)
        digits.append(digit)
        fraction -= digit
        if (fraction > 0.5 or fraction == 0.5 and digit & 1) and fraction + delta > 1:
            while digits and digits[-1] == 35:
                digits.pop()
            if digits:
                digits[-1] += 1
            break
    return "".join("0123456789abcdefghijklmnopqrstuvwxyz"[digit] for digit in digits)

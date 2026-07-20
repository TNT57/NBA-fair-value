"""NBA Fair-Value Model.

Predicts a player's *fair* salary from on-court performance only, then flags
over/underpaid players as the residual (actual - predicted). Isolation Forest /
LOF are retained as a complementary "statistically unusual player" lens.
"""

__version__ = "0.1.0"

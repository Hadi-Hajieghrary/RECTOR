"""
Trajectory Selectors
====================

Three selection methods (§6) applied to identical K=6 candidate sets:

    confidence    – Argmax of generator confidence scores (Method A)
    weighted_sum  – Per-rule weighted cost (independently tuned, Method B)
    rector_lex    – Lexicographic tier-ordered selection (Method C)

All selectors implement `BaseSelector.select()` and return a
`(selected_index, DecisionTrace)` tuple.
"""

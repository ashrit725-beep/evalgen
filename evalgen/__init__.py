"""evalgen: a live, fine-tuning-resistant LLM evaluation demo (earthquake digests).

One evaluation cycle:

    snapshot -> messy digest (+ traps) -> composed questions -> code-computed answer key
             -> models -> automatic scoring -> metamorphic consistency -> report

The method (code, templates, reading rules, scoring) is public. The instances are generated
fresh from a seed at evaluation time, so there is nothing fixed to fine-tune against.
"""

__version__ = "0.2.0"  # 0.2.0: reading rules name every notice wording, times, coordinates and magnitude scales

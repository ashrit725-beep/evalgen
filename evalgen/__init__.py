"""evalgen: a live, fine-tuning-resistant LLM evaluation demo (digests of earthquakes and, since 0.3.0,
of weather observations, near-Earth-object approaches and invented records, in seven genres).

One evaluation cycle:

    snapshot -> messy digest (+ traps) -> composed questions -> code-computed answer key
             -> models -> automatic scoring -> metamorphic consistency -> report

The method (code, templates, reading rules, scoring) is public. The instances are generated
fresh from a seed at evaluation time, so there is nothing fixed to fine-tune against.
"""

__version__ = "0.3.0"  # 0.3.0: weather, neo and invented domains, six new genres, mixes and held-out domains and genres

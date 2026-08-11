"""ridgeline: producer-agnostic label-localization QA and a gated ridge snapper for
Vesuvius Challenge segmentation labels.

Layer 0 (synthetic, exact truth) calibrates the instrument. The real-data drift claim
is scored by an INDEPENDENT witness (meijering / raw-CT), never the sato field the
snapper optimizes and never the label producer's own frangi. Layer 3 re-derives the
deterministic label geometry from its seed with no truth model. A downstream lift
probe shows a learner benefits from the correction.
"""
__version__ = "0.1.0"

from . import engine, geom, metric, snapper, synth, harness, witness, lift, data

__all__ = ["engine", "geom", "metric", "snapper", "synth", "harness",
           "witness", "lift", "data"]

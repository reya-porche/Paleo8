from .geological import build_prior_features, compute_stratigraphic_continuity_prior, flag_geological_anomaly

# Lazy import ontology to avoid requiring anthropic package
def __getattr__(name):
    if name in ("map_lith_description", "map_batch", "interpret_anomaly"):
        from . import ontology
        return getattr(ontology, name)
    raise AttributeError(f"module 'priors' has no attribute '{name}'")


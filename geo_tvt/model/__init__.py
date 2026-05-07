try:
    from .predictor       import engineer_sequence_features, train_catboost, predict_catboost, GeoTVTTransformer
    from .typewell_encoder import TypewellEncoder, encode_typewell_df
except ImportError:
    pass  # torch not available; these are only used by non-competition paths

from .trainer import train_baseline, predict_test, split_wells

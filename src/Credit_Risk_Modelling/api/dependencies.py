# src/Credit_Risk_Modelling/api/dependencies.py

from Credit_Risk_Modelling.pipeline.inference_pipeline import run_inference


def get_inference_engine():
    """
    Dependency-injected inference engine.
    Can be overridden in tests.
    """
    return run_inference

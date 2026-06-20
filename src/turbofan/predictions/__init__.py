"""Pure RUL-compute math for trained models.

This package contains framework-free functions that take a trained model
object and a data frame and compute remaining-useful-life (RUL) numbers.
It has no MLflow or FastAPI knowledge; its only consumer is
``turbofan.registry.pyfunc``, which uses it to implement MLflow
``PythonModel.predict()`` methods.
"""

"""Abstract classes for input data pipelines."""


def transform(inputs, transformations=()):  # pylint: disable=g-bare-generic
    result = inputs
    for transform_fn in transformations:
        result = transform_fn(result)
    return result

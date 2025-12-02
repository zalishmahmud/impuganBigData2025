from collections import namedtuple

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])

ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo',
    ['column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'],
)

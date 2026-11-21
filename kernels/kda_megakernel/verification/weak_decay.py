"""Supplemental packed input for KDA verification."""
from ..cases.inputs import make_inputs_from_recipe

CASE = dict(id='weak-packed', full_module=True, variable_length=True, batch=2,
                sequence_lengths=[8191, 8193], sequence=16384, hidden=256, hqk=2, hv=2,
                dk=128, dv=128, conv_width=4, dtype='bfloat16', seed=75)


def make_inputs():
    arguments = make_inputs_from_recipe(CASE)
    arguments['dt_bias_weight'].fill_(-14)
    return arguments

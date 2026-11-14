import re

import torch

from me_project_speit.utils import select_device, timestamp, to_plain


def test_timestamp_format():
    assert re.fullmatch(r"\d{8}_\d{6}", timestamp())


def test_to_plain_converts_tuples_recursively():
    assert to_plain({"a": (1, (2, 3)), "b": [(4,)]}) == {"a": [1, [2, 3]], "b": [[4]]}


def test_select_device_cpu():
    assert select_device("cpu") == torch.device("cpu")

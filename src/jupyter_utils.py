from collections import defaultdict
from typing import Union

import pandas as pd

def get_root_data(
    data: Union[pd.Series, pd.DataFrame],
    sample_index=None
) -> pd.DataFrame:

    sample = defaultdict(list)

    if isinstance(data, pd.DataFrame):
        data = data.iloc[sample_index]

    data = data.filter(regex=r"(?<!_)root_")

    for col, val in data.iteritems():
        sample[col.rsplit("_", 1)[0]].append(val)

    return pd.DataFrame(sample)
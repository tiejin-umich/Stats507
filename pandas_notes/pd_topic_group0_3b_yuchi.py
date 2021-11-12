# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using
# the exact title with spaces replaced by a dash.
#
# + [Topic Title](#Topic-Title)
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide.

# ## Pandas Topic- Windows operations.
# *Tiejin Chen*; **tiejin@umich.edu**
# Here we choose windows operations(```rolling```) as our topic.
# First, we import all the module we need
#

import numpy as np
import pandas as pd

# ## Windows operation
# - In the region of data science, sometimes we need to manipulate
# one raw with two raws next to it for every raw.
# - this is one kind of windows operation.
# - We define windows operation as an operation that
# performs an aggregation over a sliding partition of values (from pandas' userguide)
# - Using ```df.rolling``` function to use the normal windows operation

rng = np.random.default_rng(9 * 2021 * 20)
n=5
a = rng.binomial(n=1, p=0.5, size=n)
b = 1 - 0.5 * a + rng.normal(size=n)
c = 0.8 * a + rng.normal(size=n)
df = pd.DataFrame({'a': a, 'b': b, 'c': c})
print(df)
df['b'].rolling(window=2).sum()

# ## Rolling parameter
# In ```rolling``` method, we have some parameter to control the method, And we introduce two:
# - center: Type bool; if center is True, Then the result will move to the center in series.
# - window: decide the length of window or the customed window

df['b'].rolling(window=3).sum()

df['b'].rolling(window=3,center=True).sum()

df['b'].rolling(window=2).sum()

# +
# example of customed window
window_custom = [True,False,True,False,True]
from pandas.api.indexers import BaseIndexer
class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            if self.use_expanding[i]:
                start[i] = 0
                end[i] = i + 1
            else:
                start[i] = i
                end[i] = i + self.window_size
        return start, end
indexer1 = CustomIndexer(window_size=1, use_expanding=window_custom)
indexer2 = CustomIndexer(window_size=2, use_expanding=window_custom)

df['b'].rolling(window=indexer1).sum()
# -

df['b'].rolling(window=indexer2).sum()

# ## Windows operation with groupby
# - ```pandas.groupby``` type also have windows operation method,
# Hence we can combine groupby and windows operation.
# - we can also use ```apply``` after we use ```rolling```

df.groupby('a').rolling(window=2).sum()


def test_mean(x):
    return x.mean()
df['b'].rolling(window=2).apply(test_mean)

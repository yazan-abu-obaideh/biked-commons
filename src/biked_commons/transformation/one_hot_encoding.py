from typing import Callable, List
import numpy as np
import pandas as pd

# columns to one‐hot encode
ONE_HOT_ENCODED_CLIPS_COLUMNS: List[str] = [
    'MATERIAL',
    'Dropout spacing style',
    'Head tube type',
    'RIM_STYLE front',
    'RIM_STYLE rear',
    'Handlebar style',
    'Stem kind',
    'Fork type',
    'Seat tube type',
]

# columns that are already boolean and should stay in the DF (converted to float on encode)
BOOLEAN_COLUMNS: List[str] = [
    'bottle SEATTUBE0 show',
    'bottle DOWNTUBE0 show',
    'BELTorCHAIN',
    'SSB_Include',
    'CSB_Include',
]

PREFIX_SEP = " OHCLASS: "


def encode_to_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‐hot–encode the categorical columns in ONE_HOT_ENCODED_CLIPS_COLUMNS
    using prefix "<col> OHCLASS: <category>".  Leave all other columns
    (including BOOLEAN_COLUMNS) in place, but convert the booleans to floats.
    """
    out = df.copy(deep=True)

    # 1) one‐hot encode each categorical column
    for col in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        dummies = pd.get_dummies(
            out[col].astype(str),
            prefix=col,
            prefix_sep=PREFIX_SEP
        )
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)

    # 2) convert boolean columns to floats
    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float)
    out = out.astype(np.float32)
    return out


def decode_to_mixed(encoded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse the one‐hot encoding done by encode_clips:
    - For each original categorical column, find all "<col> OHCLASS: *" dummies,
      take argmax (the position of the 1), strip off the prefix, and restore
      the category string.
    - Round the float boolean columns back to 0/1 and cast to bool.
    """
    out = encoded_df.copy(deep=True)

    # 1) decode each categorical variable
    for col in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        pref = f"{col}{PREFIX_SEP}"
        # gather the dummy cols for this variable
        dummy_cols = [c for c in out.columns if c.startswith(pref)]
        if not dummy_cols:
            continue

        # idxmax gives the column name with the highest value (i.e., the 1)
        restored = (
            out[dummy_cols]
            .astype(int)
            .idxmax(axis=1)
            .str.replace(pref, "", n=1, regex=False)
        )

        out[col] = restored
        out.drop(columns=dummy_cols, inplace=True)

    # 2) round boolean floats back to bool
    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = out[col].round().astype(int).astype(bool)

    return out


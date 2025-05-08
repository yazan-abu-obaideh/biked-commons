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

ALL_CATEGORIES = {
    'MATERIAL': [
        'ALUMINIUM',
        'BAMBOO',
        'CARBON',
        'OTHER',
        'STEEL',
        'TITANIUM'
    ],
    'Dropout spacing style': [
        '0',
        '1',
        '2',
        '3'
    ],
    'Head tube type': [
        '0',
        '1',
        '2',
        '3'
    ],
    'RIM_STYLE front': [
        'DISC',
        'SPOKED',
        'TRISPOKE'
    ],
    'RIM_STYLE rear': [
        'DISC',
        'SPOKED',
        'TRISPOKE'
    ],
    'Handlebar style': [
        '0',
        '1',
        '2'
    ],
    'Stem kind': [
        '0',
        '1',
        '2'
    ],
    'Fork type': [
        '0',
        '1',
        '2'
    ],
    'Seat tube type': [
        '0',
        '1',
        '2'
    ]
}


# columns that are already boolean and should stay in the DF (converted to float on encode)
BOOLEAN_COLUMNS: List[str] = [
    'bottle SEATTUBE0 show',
    'bottle DOWNTUBE0 show',
    'BELTorCHAIN',
    'SEATSTAYbrdgCheck',
    'CHAINSTAYbrdgCheck',
]

FAKE_BOOLEAN_COLUMNS: List[str] = ['BELTorCHAIN']

PREFIX_SEP = " OHCLASS: "


def encode_to_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‐hot–encode the categorical columns in ONE_HOT_ENCODED_CLIPS_COLUMNS
    using prefix "<col> OHCLASS: <category>".  Leave all other columns
    (including BOOLEAN_COLUMNS) in place, but convert the booleans to floats.
    This function ensures that all possible categories are included, even if some
    are missing in the current slice.
    """
    out = df.copy(deep=True)

    # 1) One-hot encode each categorical column
    for col in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        # Get all possible categories from the ALL_CATEGORIES dictionary
        all_categories = ALL_CATEGORIES.get(col, [])
        
        # Create dummy variables for the current slice of data
        dummies = pd.get_dummies(
            out[col].astype(str),
            prefix=col,
            prefix_sep=PREFIX_SEP
        )

        # Ensure all categories are represented, even if missing
        for category in all_categories:
            category_col = f"{col}{PREFIX_SEP}{category}"
            if category_col not in dummies.columns:
                dummies[category_col] = 0

        # Reorder columns to match all possible categories
        dummies = dummies[sorted(dummies.columns)]

        # Drop the original column and concatenate the dummies
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)

    # 2) Convert boolean columns to floats
    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float)

    out = out.astype(np.float32)
    return out



def decode_to_mixed(encoded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reverse the one‐hot encoding done by encode_clips:
    - For each original categorical column, find all "<col> OHCLASS: *" dummies,
      take argmax (the position of the highest value), strip off the prefix, and restore
      the category string.
    - Round the float boolean columns back to 0/1 and cast to bool.
    """
    out = encoded_df.copy(deep=True)

    # 1) decode each categorical variable
    for col in ONE_HOT_ENCODED_CLIPS_COLUMNS:
        pref = f"{col}{PREFIX_SEP}"
        dummy_cols = [c for c in out.columns if c.startswith(pref)]
        if not dummy_cols:
            continue

        # idxmax on the raw floats picks the column with the highest value
        restored = (
            out[dummy_cols]
            .idxmax(axis=1)
            .str.replace(pref, "", n=1, regex=False)
        )

        out[col] = restored
        out.drop(columns=dummy_cols, inplace=True)

    # 2) round boolean floats back to bool
    for col in BOOLEAN_COLUMNS:
        if col in out.columns and col not in FAKE_BOOLEAN_COLUMNS:
            out[col] = out[col].round().astype(int).astype(bool)

    return out



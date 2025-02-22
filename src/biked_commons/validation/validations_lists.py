__MULTIPLIER = 1000

# TODO: grab all validations from biked/functions

__COMBINED_VALIDATIONS_RAW = [
    lambda df: df["Saddle height"] < (df["ST Length"] * __MULTIPLIER) + 40,
    lambda df: df["Saddle height"] > ((df["ST Length"] * __MULTIPLIER) + df["Seatpost LENGTH"] + 30),
    lambda df: df["BSD rear"] < df["ERD rear"],
    lambda df: df["BSD front"] < df["ERD front"],
    lambda df: df["HT LX"] >= df["HT Length"],
    lambda df: ((df["HT UX"] + df["HT LX"]) >= df['HT Length']),
]

import pandas as pd

from biked_commons.combined_representation.combined_representation import CombinedRepresentation, DatasetDescription


class BikeRepresentations:
    FRAMED = "framed"
    CLIP = "clip"
    BIKE_FIT = "bike_fit"


ORIGINAL = CombinedRepresentation(
    id_to_description={
        BikeRepresentations.FRAMED: DatasetDescription(data=pd.DataFrame(), conversions=[]),
        BikeRepresentations.CLIP: DatasetDescription(data=pd.DataFrame(), conversions=[]),
        BikeRepresentations.BIKE_FIT: DatasetDescription(data=pd.DataFrame(), conversions=[]),
    }
)

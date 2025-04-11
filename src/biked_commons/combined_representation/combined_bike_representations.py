import pandas as pd

from biked_commons.combined_representation.combined_representation import CombinedRepresentation, DatasetDescription
from biked_commons.combined_representation.conversions import RenameColumn, ScaleColumn
from biked_commons.combined_representation.load_data import load_augmented_framed_dataset
from biked_commons.resource_utils import resource_path


class BikeRepresentations:
    FRAMED = "framed"
    CLIP = "clip"
    BIKE_FIT = "bike_fit"


__MASTER_TO_CLIPS_M_TO_MM = {'SS OD': 'ssd', 'HT Length': 'Head tube length textfield', 'CS OD': 'csd',
                             'ST UX': 'Seat tube extension2', 'HT LX': 'Head tube lower extension2',
                             'HT UX': 'Head tube upper extension2', 'ST Length': 'Seat tube length',
                             'BB Drop': 'BB textfield', 'CSB Offset': 'CHAINSTAYbrdgshift',
                             'SS E': 'Seat stay junction0',
                             'Dropout Offset': 'Dropout spacing', 'SS Z': 'SSTopZOFFSET', 'Stack': 'Stack',
                             'CS Length': 'CS textfield', 'DT Length': 'DT Length', 'HT OD': 'Head tube diameter',
                             'SSB Offset': 'SEATSTAYbrdgshift', 'BB OD': 'BB diameter', 'DT OD': 'dtd', 'TT OD': 'ttd',
                             'BB Length': 'BB length', 'ST OD': 'Seat tube diameter'}


def _framed_description():
    x_framed, y_framed, x_scaler, y_scaler = load_augmented_framed_dataset()
    framed = pd.DataFrame(x_scaler.inverse_transform(x_framed), columns=x_framed.columns, index=x_framed.index)
    return DatasetDescription(
        data=framed,
        conversions=[]
    )


def _clip_description():
    clips = pd.read_csv(resource_path('datasets/raw_datasets/clip_sBIKED_processed.csv'), index_col=0)
    clips.index = [str(idx) for idx in clips.index]
    conversions = []
    for master, clip in __MASTER_TO_CLIPS_M_TO_MM.items():
        conversions.append(ScaleColumn(
            column=clip,
            multiplier=(1 / 1000)
        ))
        conversions.append(RenameColumn(
            from_name=clip,
            to_name=master
        ))
    return DatasetDescription(
        data=clips,
        conversions=conversions
    )


def _bike_fit_description():
    bike_fit = (pd.read_csv(resource_path('datasets/raw_datasets/bike_vector_df_with_id.csv'),
                            index_col='Bike ID')
                .drop(columns=['Unnamed: 0']))
    bike_fit.index = [str(idx) for idx in bike_fit.index]
    return DatasetDescription(
        data=bike_fit,
        conversions=[]
    )


COMBINED = CombinedRepresentation(
    id_to_description={
        BikeRepresentations.FRAMED: _framed_description(),
        BikeRepresentations.CLIP: _clip_description(),
        BikeRepresentations.BIKE_FIT: _bike_fit_description(),
    }
)

import pandas as pd

from biked_commons.validation.base_validation_function import ValidationFunction


class DisallowedNegativeValues(ValidationFunction):
    def friendly_name(self) -> str:
        return "Disallowed negative values"

    def validate(self, designs: pd.DataFrame) -> pd.DataFrame:
        raise Exception("Unimplemented function!")


# TODO: add FDERD, RDERD, FDBSD, RDBSD, CS textfield to that list
_MANDATORY_POSITIVE = ['CS textfield', 'Stack', 'Head angle',
                       'Head tube length textfield', 'Seat tube length',
                       'Seat angle', 'DT Length', 'BB diameter', 'ttd', 'dtd', 'csd', 'ssd',
                       'Chain stay position on BB', 'MATERIAL',
                       'Head tube upper extension2', 'Seat tube extension2',
                       'Head tube lower extension2', 'SEATSTAYbrdgshift', 'CHAINSTAYbrdgshift',
                       'SEATSTAYbrdgdia1', 'CHAINSTAYbrdgdia1', 'SEATSTAYbrdgCheck',
                       'CHAINSTAYbrdgCheck', 'Dropout spacing',
                       'Wall thickness Bottom Bracket', 'Wall thickness Top tube',
                       'Wall thickness Head tube', 'Wall thickness Down tube',
                       'Wall thickness Chain stay', 'Wall thickness Seat stay',
                       'Wall thickness Seat tube', 'ERD rear', 'Wheel width rear',
                       'Dropout spacing style', 'BSD front', 'Wheel width front', 'ERD front',
                       'BSD rear', 'Fork type', 'Stem kind', 'Display AEROBARS',
                       'Handlebar style', 'Head tube type', 'BB length', 'Head tube diameter',
                       'Wheel cut', 'Seat tube diameter', 'Top tube type',
                       'bottle SEATTUBE0 show', 'bottle DOWNTUBE0 show',
                       'Front Fender include', 'Rear Fender include', 'BELTorCHAIN',
                       'Number of cogs', 'Number of chainrings', 'Display RACK',
                       'FIRST color R_RGB', 'FIRST color G_RGB', 'FIRST color B_RGB',
                       'RIM_STYLE front', 'RIM_STYLE rear', 'SPOKES composite front',
                       'SBLADEW front', 'SBLADEW rear', 'Saddle length', 'Saddle height',
                       'Down tube diameter', 'Seatpost LENGTH']

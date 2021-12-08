import typing as t

import numpy as np
import pandas as pd  # type: ignore
from marshmallow import Schema, ValidationError, fields

from fraud_detection_model.config.core import config


def merge_datasets(
    *, transaction: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    dataframe = pd.merge(transaction, identity, how="left", on=config.model_config.id)
    return dataframe


def validate_inputs(
    *, transaction: pd.DataFrame, identity: pd.DataFrame
) -> t.Tuple[t.Any, t.Union[t.List[str], t.List[t.Any], t.Dict[t.Any, t.Any], None]]:
    """Check model inputs for unprocessable values."""

    # merge datasets together on TransactionID
    dataset = merge_datasets(transaction=transaction, identity=identity)
    validated_data = dataset.rename(columns=config.model_config.test_features_to_rename)

    # set many=True to allow passing in a list
    schema = TransactionAndIdentityInputData(many=True)
    errors = None

    try:
        # replace numpy nans so that Marshmallow can validate
        data_ = validated_data.replace({np.nan: None}).to_dict(orient="records")
        schema.load(data_)
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors


class TransactionAndIdentityInputData(Schema):
    V13 = fields.Float(allow_none=True)
    V14 = fields.Float(allow_none=True)
    V15 = fields.Float(allow_none=True)
    V16 = fields.Float(allow_none=True)
    V17 = fields.Float(allow_none=True)
    V18 = fields.Float(allow_none=True)
    V19 = fields.Float(allow_none=True)
    V27 = fields.Float(allow_none=True)
    V28 = fields.Float(allow_none=True)
    V32 = fields.Float(allow_none=True)
    V98 = fields.Float(allow_none=True)
    V116 = fields.Float(allow_none=True)
    V117 = fields.Float(allow_none=True)
    V118 = fields.Float(allow_none=True)
    V119 = fields.Float(allow_none=True)
    V120 = fields.Float(allow_none=True)
    V297 = fields.Float(allow_none=True)
    V300 = fields.Float(allow_none=True)
    V301 = fields.Float(allow_none=True)
    V325 = fields.Float(allow_none=True)
    V328 = fields.Float(allow_none=True)
    TransactionDT = fields.Integer(allow_none=True)
    TransactionAmt = fields.Float(allow_none=True)
    C1 = fields.Float(allow_none=True)
    C2 = fields.Float(allow_none=True)
    C3 = fields.Float(allow_none=True)
    C4 = fields.Float(allow_none=True)
    C5 = fields.Float(allow_none=True)
    C6 = fields.Float(allow_none=True)
    C7 = fields.Float(allow_none=True)
    C8 = fields.Float(allow_none=True)
    C9 = fields.Float(allow_none=True)
    C10 = fields.Float(allow_none=True)
    C11 = fields.Float(allow_none=True)
    C12 = fields.Float(allow_none=True)
    C13 = fields.Float(allow_none=True)
    C14 = fields.Float(allow_none=True)
    V97 = fields.Float(allow_none=True)
    V99 = fields.Float(allow_none=True)
    V100 = fields.Float(allow_none=True)
    V101 = fields.Float(allow_none=True)
    V102 = fields.Float(allow_none=True)
    V126 = fields.Float(allow_none=True)
    V127 = fields.Float(allow_none=True)
    V153 = fields.Float(allow_none=True)
    V154 = fields.Float(allow_none=True)
    V157 = fields.Float(allow_none=True)
    V158 = fields.Float(allow_none=True)
    V166 = fields.Float(allow_none=True)
    V293 = fields.Float(allow_none=True)
    V294 = fields.Float(allow_none=True)
    V306 = fields.Float(allow_none=True)
    V307 = fields.Float(allow_none=True)
    V308 = fields.Float(allow_none=True)
    V310 = fields.Float(allow_none=True)
    V311 = fields.Float(allow_none=True)
    V312 = fields.Float(allow_none=True)
    V313 = fields.Float(allow_none=True)
    V316 = fields.Float(allow_none=True)
    V317 = fields.Float(allow_none=True)
    V318 = fields.Float(allow_none=True)
    V319 = fields.Float(allow_none=True)
    V320 = fields.Float(allow_none=True)
    V321 = fields.Float(allow_none=True)
    V324 = fields.Float(allow_none=True)
    V326 = fields.Float(allow_none=True)
    V327 = fields.Float(allow_none=True)
    V329 = fields.Float(allow_none=True)
    V330 = fields.Float(allow_none=True)
    V331 = fields.Float(allow_none=True)
    V332 = fields.Float(allow_none=True)
    V336 = fields.Float(allow_none=True)
    R_emaildomain = fields.Str(allow_none=True)
    card1 = fields.Float(allow_none=True)
    card2 = fields.Float(allow_none=True)
    card3 = fields.Float(allow_none=True)
    card5 = fields.Float(allow_none=True)
    addr1 = fields.Float(allow_none=True)
    addr2 = fields.Float(allow_none=True)
    id_13 = fields.Float(allow_none=True)
    id_17 = fields.Float(allow_none=True)
    id_19 = fields.Float(allow_none=True)
    id_20 = fields.Float(allow_none=True)
    id_21 = fields.Float(allow_none=True)
    id_26 = fields.Float(allow_none=True)
    ProductCD = fields.Str(allow_none=True)
    TransactionID = fields.Integer(allow_none=True)
    V95 = fields.Float(allow_none=True)
    V96 = fields.Float(allow_none=True)
    V145 = fields.Float(allow_none=True)
    V235 = fields.Float(allow_none=True)
    V279 = fields.Float(allow_none=True)
    V280 = fields.Float(allow_none=True)
    V284 = fields.Float(allow_none=True)
    V285 = fields.Float(allow_none=True)
    V286 = fields.Float(allow_none=True)
    V287 = fields.Float(allow_none=True)
    V290 = fields.Float(allow_none=True)
    V291 = fields.Float(allow_none=True)
    V292 = fields.Float(allow_none=True)
    V295 = fields.Float(allow_none=True)
    V298 = fields.Float(allow_none=True)
    V299 = fields.Float(allow_none=True)
    V302 = fields.Float(allow_none=True)
    V303 = fields.Float(allow_none=True)
    V304 = fields.Float(allow_none=True)
    V305 = fields.Float(allow_none=True)
    V309 = fields.Float(allow_none=True)
    V322 = fields.Float(allow_none=True)
    V323 = fields.Float(allow_none=True)
    V333 = fields.Float(allow_none=True)
    V334 = fields.Float(allow_none=True)
    V335 = fields.Float(allow_none=True)
    V337 = fields.Float(allow_none=True)
    V338 = fields.Float(allow_none=True)
    V339 = fields.Float(allow_none=True)
    id_08 = fields.Float(allow_none=True)
    id_16 = fields.Str(allow_none=True)
    id_27 = fields.Str(allow_none=True)
    DeviceInfo = fields.Str(allow_none=True)

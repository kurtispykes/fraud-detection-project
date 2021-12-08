import typing as t

import numpy as np
import pandas as pd  # type: ignore
from pydantic import BaseModel, ValidationError

from fraud_detection_model.config.core import config


def merge_datasets(
    *, transaction: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    dataframe = pd.merge(transaction, identity, how="left", on=config.model_config.id)
    return dataframe


def validate_inputs(
    *, transaction: pd.DataFrame, identity: pd.DataFrame
) -> t.Tuple[t.Any, t.Optional[str]]:
    """Check model inputs for unprocessable values."""

    # merge datasets together on TransactionID
    dataset = merge_datasets(transaction=transaction, identity=identity)
    validated_data = dataset.rename(columns=config.model_config.test_features_to_rename)

    # replace numpy nans so that Marshmallow can validate
    data_ = validated_data.replace({np.nan: None}).to_dict(orient="records")
    errors = None

    try:
        MultipleTransactionAndIdentityInputData(inputs=data_)
    except ValidationError as exc:
        errors = exc.json()

    return validated_data, errors


class TransactionAndIdentityInputData(BaseModel):
    V13: t.Optional[float]
    V14: t.Optional[float]
    V15: t.Optional[float]
    V16: t.Optional[float]
    V17: t.Optional[float]
    V18: t.Optional[float]
    V19: t.Optional[float]
    V27: t.Optional[float]
    V28: t.Optional[float]
    V32: t.Optional[float]
    V98: t.Optional[float]
    V116: t.Optional[float]
    V117: t.Optional[float]
    V118: t.Optional[float]
    V119: t.Optional[float]
    V120: t.Optional[float]
    V297: t.Optional[float]
    V300: t.Optional[float]
    V301: t.Optional[float]
    V325: t.Optional[float]
    V328: t.Optional[float]
    TransactionDT: t.Optional[int]
    TransactionAmt: t.Optional[float]
    C1: t.Optional[float]
    C2: t.Optional[float]
    C3: t.Optional[float]
    C4: t.Optional[float]
    C5: t.Optional[float]
    C6: t.Optional[float]
    C7: t.Optional[float]
    C8: t.Optional[float]
    C9: t.Optional[float]
    C10: t.Optional[float]
    C11: t.Optional[float]
    C12: t.Optional[float]
    C13: t.Optional[float]
    C14: t.Optional[float]
    V97: t.Optional[float]
    V99: t.Optional[float]
    V100: t.Optional[float]
    V101: t.Optional[float]
    V102: t.Optional[float]
    V126: t.Optional[float]
    V127: t.Optional[float]
    V153: t.Optional[float]
    V154: t.Optional[float]
    V157: t.Optional[float]
    V158: t.Optional[float]
    V166: t.Optional[float]
    V293: t.Optional[float]
    V294: t.Optional[float]
    V306: t.Optional[float]
    V307: t.Optional[float]
    V308: t.Optional[float]
    V310: t.Optional[float]
    V311: t.Optional[float]
    V312: t.Optional[float]
    V313: t.Optional[float]
    V316: t.Optional[float]
    V317: t.Optional[float]
    V318: t.Optional[float]
    V319: t.Optional[float]
    V320: t.Optional[float]
    V321: t.Optional[float]
    V324: t.Optional[float]
    V326: t.Optional[float]
    V327: t.Optional[float]
    V329: t.Optional[float]
    V330: t.Optional[float]
    V331: t.Optional[float]
    V332: t.Optional[float]
    V336: t.Optional[float]
    R_emaildomain: t.Optional[str]
    card1: t.Optional[float]
    card2: t.Optional[float]
    card3: t.Optional[float]
    card5: t.Optional[float]
    addr1: t.Optional[float]
    addr2: t.Optional[float]
    id_13: t.Optional[float]
    id_17: t.Optional[float]
    id_19: t.Optional[float]
    id_20: t.Optional[float]
    id_21: t.Optional[float]
    id_26: t.Optional[float]
    ProductCD: t.Optional[str]
    TransactionID: t.Optional[int]
    V95: t.Optional[float]
    V96: t.Optional[float]
    V145: t.Optional[float]
    V235: t.Optional[float]
    V279: t.Optional[float]
    V280: t.Optional[float]
    V284: t.Optional[float]
    V285: t.Optional[float]
    V286: t.Optional[float]
    V287: t.Optional[float]
    V290: t.Optional[float]
    V291: t.Optional[float]
    V292: t.Optional[float]
    V295: t.Optional[float]
    V298: t.Optional[float]
    V299: t.Optional[float]
    V302: t.Optional[float]
    V303: t.Optional[float]
    V304: t.Optional[float]
    V305: t.Optional[float]
    V309: t.Optional[float]
    V322: t.Optional[float]
    V323: t.Optional[float]
    V333: t.Optional[float]
    V334: t.Optional[float]
    V335: t.Optional[float]
    V337: t.Optional[float]
    V338: t.Optional[float]
    V339: t.Optional[float]
    id_08: t.Optional[float]
    id_16: t.Optional[str]
    id_27: t.Optional[str]
    DeviceInfo: t.Optional[str]


class MultipleTransactionAndIdentityInputData(BaseModel):
    inputs: t.List[TransactionAndIdentityInputData]

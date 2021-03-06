from typing import Any, List, Optional

from fraud_detection_model.processing import validation as v  # type: ignore
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultipleTransactionAndIdentityData(BaseModel):
    inputs: List[v.TransactionAndIdentityInputData]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "TransactionID": 3663549,
                        "TransactionDT": 18403224,
                        "TransactionAmt": 31.95,
                        "ProductCD": "W",
                        "card1": 10409,
                        "card2": 111.0,
                        "card3": 150.0,
                        "card5": 226.0,
                        "addr1": 170.0,
                        "addr2": 87.0,
                        "R_emaildomain": None,
                        "C1": 6.0,
                        "C2": 6.0,
                        "C3": 0.0,
                        "C4": 0.0,
                        "C5": 3.0,
                        "C6": 4.0,
                        "C7": 0.0,
                        "C8": 0.0,
                        "C9": 6.0,
                        "C10": 5.0,
                        "C11": 1.0,
                        "C12": 115.0,
                        "C13": 6.0,
                        "C14": 0.0,
                        "V13": 1.0,
                        "V14": 0.0,
                        "V15": 0.0,
                        "V16": 0.0,
                        "V17": 0.0,
                        "V18": 0.0,
                        "V19": 0.0,
                        "V27": 0.0,
                        "V28": 0.0,
                        "V32": 0.0,
                        "V95": 1.0,
                        "V96": 0.0,
                        "V97": 0.0,
                        "V98": 1.0,
                        "V99": 1.0,
                        "V100": 0.0,
                        "V101": 0.0,
                        "V102": 0.0,
                        "V116": 1.0,
                        "V117": 1.0,
                        "V118": 1.0,
                        "V119": 1.0,
                        "V120": 1.0,
                        "V126": 0.0,
                        "V127": 47.95000076293945,
                        "V145": None,
                        "V153": None,
                        "V154": None,
                        "V158": None,
                        "V166": None,
                        "V235": None,
                        "V279": None,
                        "V280": 0.0,
                        "V284": 0.0,
                        "V285": 0.0,
                        "V286": 1.0,
                        "V287": 0.0,
                        "V290": 0.0,
                        "V291": 1.0,
                        "V292": 1.0,
                        "V293": 1.0,
                        "V294": 0.0,
                        "V295": 0.0,
                        "V297": 0.0,
                        "V298": 0.0,
                        "V299": 0.0,
                        "V300": 0.0,
                        "V301": 0.0,
                        "V302": 0.0,
                        "V303": 0.0,
                        "V304": 0.0,
                        "V305": 1.0,
                        "V306": 0.0,
                        "V307": 47.95000076293945,
                        "V308": 0.0,
                        "V309": 0.0,
                        "V310": 47.95000076293945,
                        "V311": 0.0,
                        "V312": 0.0,
                        "V313": 0.0,
                        "V316": 0.0,
                        "V317": 0.0,
                        "V318": 0.0,
                        "V319": 0.0,
                        "V320": 0.0,
                        "V321": 0.0,
                        "V322": None,
                        "V323": None,
                        "V324": None,
                        "V325": None,
                        "V326": None,
                        "V327": None,
                        "V328": None,
                        "V329": None,
                        "V330": None,
                        "V331": None,
                        "V332": None,
                        "V333": None,
                        "V334": None,
                        "V335": None,
                        "V336": None,
                        "V337": None,
                        "V338": None,
                        "V339": None,
                        "V339": None,
                        "id-08": None,
                        "id-13": None,
                        "id-16": None,
                        "id-17": None,
                        "id-19": None,
                        "id-20": None,
                        "id-21": None,
                        "id-26": None,
                        "id-27": None,
                        "DeviceInfo": None,
                    }
                ]
            }
        }

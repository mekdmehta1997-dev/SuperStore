import pandas as pd
import os

from datetime import datetime

from src.app_config import LOG_FILE

# ==================================================
# CREATE LOG DIRECTORY
# ==================================================
os.makedirs("logs", exist_ok=True)

# ==================================================
# CREATE CSV IF NOT EXISTS
# ==================================================
if not os.path.exists(LOG_FILE):

    columns = [
        "Timestamp",
        "Quantity",
        "Sales",
        "Profit",
        "Region",
        "Category",
        "Sub_Category",
        "Ship_Mode",
        "Segment",
        "City",
        "State",
        "Predicted_Discount"
    ]

    pd.DataFrame(columns=columns).to_csv(
        LOG_FILE,
        index=False
    )

# ==================================================
# LOG FUNCTION
# ==================================================
def log_prediction(data, prediction):

    log_data = {
        "Timestamp": datetime.now(),

        "Quantity": data["Quantity"],
        "Sales": data["Sales"],
        "Profit": data["Profit"],

        "Region": data["Region"],
        "Category": data["Category"],

        "Sub_Category": data["Sub-Category"],
        "Ship_Mode": data["Ship Mode"],

        "Segment": data["Segment"],
        "City": data["City"],
        "State": data["State"],

        "Predicted_Discount": prediction
    }

    df = pd.DataFrame([log_data])

    df.to_csv(
        LOG_FILE,
        mode='a',
        header=False,
        index=False
    )
from fastapi import FastAPI
from fastapi import Form
from fastapi import Request

from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates

from src.predict import predict_discount

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# ==================================================
# HOME PAGE
# ==================================================
@app.get("/", response_class=HTMLResponse)

async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# ==================================================
# PREDICTION ROUTE
# ==================================================
@app.post("/predict", response_class=HTMLResponse)

async def predict(
    request: Request,

    Quantity: int = Form(...),
    Sales: float = Form(...),
    Profit: float = Form(...),

    Region: str = Form(...),
    Category: str = Form(...),

    Sub_Category: str = Form(...),
    Ship_Mode: str = Form(...),

    Segment: str = Form(...),
    City: str = Form(...),
    State: str = Form(...)
):

    data = {

        'Quantity': Quantity,
        'Sales': Sales,
        'Profit': Profit,

        'Region': Region,
        'Category': Category,

        'Sub-Category': Sub_Category,
        'Ship Mode': Ship_Mode,

        'Segment': Segment,
        'City': City,
        'State': State
    }

    prediction = predict_discount(data)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(prediction, 4)
        }
    )
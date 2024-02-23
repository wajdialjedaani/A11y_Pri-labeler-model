from fastapi import Depends, FastAPI
from .classifier.model import Model, get_model

app = FastAPI()


@app.post("/predict")
async def predict(issue: str, model: Model = Depends(get_model)):
    bug_confidence, improvement_confidence, newFeature_confidence = model.predict(
        issue)
    return {"Bug": str(bug_confidence),
            "Improvement": str(improvement_confidence),
            "New Feature": str(newFeature_confidence)}

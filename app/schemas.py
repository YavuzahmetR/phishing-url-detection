from pydantic import BaseModel, Field

class URLFeaturesInput(BaseModel):
    DomainLength: int = Field(..., example=15)
    IsDomainIP: int = Field(..., example=0)
    TLDLength: int = Field(..., example=3)
    NoOfSubDomain: int = Field(..., example=1)
    IsHTTPS: int = Field(..., example=1)
    CharContinuationRate: float = Field(..., example=0.85)
    HasObfuscation: int = Field(..., example=0)
    ObfuscationRatio: float = Field(..., example=0.0)
    LetterRatioInURL: float = Field(..., example=0.75)
    DegitRatioInURL: float = Field(..., example=0.05)
    SpacialCharRatioInURL: float = Field(..., example=0.20)
    NoOfEqualsInURL: int = Field(..., example=0)
    NoOfQMarkInURL: int = Field(..., example=0)
    NoOfAmpersandInURL: int = Field(..., example=0)

class PredictionOutput(BaseModel):
    is_phishing: bool
    threat_label: str
    confidence_score: float
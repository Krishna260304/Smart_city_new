from pydantic import BaseModel
class IssueIn(BaseModel):
    description: str
    latitude: float
    longitude: float
    image: str
    severity: str
    deviceId: str | None = None
    source: str | None = None
    scope: str | None = None

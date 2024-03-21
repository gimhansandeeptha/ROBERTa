from fastapi import APIRouter
from ..database.main import Database

router = APIRouter()

@router.get("/get_comments_with_sentiment",tags=['api'])
async def get_comments_with_sentiment():
    db = Database()
    cases = db.get_cases_by_date(start_date="2024-03-19", end_date="2024-03-19")
    return cases

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Attachment(BaseModel):
    path: str
    sys_id: str
    size_bytes: int
    content_type: str
    size: str
    file_name: str
    state: str


class Entry(BaseModel):
    sys_created_on_adjusted: str
    sys_id: str
    login_name: str
    user_sys_id: str
    initials: str
    sys_created_on: str
    contains_code: Optional[str] = None
    field_label: Optional[str] = None
    name: str
    value: Optional[str] = None
    element: str
    attachment: Optional[Attachment] = None


class JournalField(BaseModel):
    can_read: bool
    color: str
    can_write: bool
    name: str
    label: str


class Case(BaseModel):
    display_value: str
    sys_id: str
    short_description: str
    number: str
    entries: List[Entry]
    user_sys_id: str
    user_full_name: str
    user_login: str
    label: str
    table: str
    journal_fields: List[JournalField]
    sys_created_on: str
    sys_created_on_adjusted: str
    account: str

class Result(BaseModel):
    cases: List[Case]


class Model(BaseModel):
    result: Result

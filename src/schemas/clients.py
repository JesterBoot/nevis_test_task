from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class ClientCreate(BaseModel):
    first_name: str = Field(min_length=1, max_length=255)
    last_name: str = Field(min_length=1, max_length=255)
    email: EmailStr
    country_of_residence: str | None = Field(
        default=None,
        alias="countryOfResidence",
        max_length=255,
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("email", mode="before")
    @classmethod
    def strip_email(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class ClientResponse(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    email: EmailStr
    country_of_residence: str | None = Field(
        default=None,
        alias="countryOfResidence",
    )

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )


__all__ = ("ClientCreate", "ClientResponse")

from typing import TypedDict, Literal

from pydantic import BaseModel

from docassist.parametrized import expand_on, cross_product

AudienceRole = Literal["enduser", "maintainer"]
AudienceToProjectRelationship = Literal["user", "developer"]

class Description[T](TypedDict):
    slug: T
    details: str

class ContentDerivationPerspective(TypedDict):
    role: Description[AudienceRole]
    relationship_to_project: Description[AudienceToProjectRelationship]
    summary: str

ROLES = {
    "enduser": {
        "slug": "enduser",
        "details": "someone who uses the project we're working on to solve the problems or finish some tasks"
    },
    "maintainer": {
        "slug": "maintainer",
        "details": "someone who work with the project, but doesn't use it directly; someone that makes sure that "
                   "endusers can use it"
    }
}

LITERACY = {
    "user": {
        "slug": "user",
        "details": "someone that can make use of features of this project, but is not working on this project directly"
    },
    "developer": {
        "slug": "developer",
        "details": "TODO"
    }
}

SUMMARY = {
    ("enduser", "user"): "...",
    ("enduser", "developer"): "...",
    ("maintainer", "user"): "...",
    ("maintainer", "developer"): "...",
}

class PerspectivePointer(BaseModel):
    role:AudienceRole
    relationship_to_project: AudienceToProjectRelationship

def perspective(role: AudienceRole, relationship_to_project: AudienceToProjectRelationship) -> ContentDerivationPerspective:
    return {
        "role": ROLES[role],
        "relationship_to_project": LITERACY[relationship_to_project],
        "summary": SUMMARY[(role, relationship_to_project)]
    }

PERSPECTIVES = list(
    cross_product(
        {
            "role": expand_on(AudienceRole),
            "relationship_to_project": expand_on(AudienceToProjectRelationship)
        }
    )
)

FINAL_DOCUMENTATION_PERSPECTIVE_POINTER = dict(role="enduser", relationship_to_project="user")
FINAL_DOCUMENTATION_PERSPECTIVE = perspective(**FINAL_DOCUMENTATION_PERSPECTIVE_POINTER)
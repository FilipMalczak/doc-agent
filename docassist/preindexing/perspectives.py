from typing import TypedDict, Literal

from docassist.structured_agent import cross_product, expand_on

AudienceRole = Literal["end-user", "maintainer"]
AudienceToProjectRelationship = Literal["user", "developer"]

class Description[T](TypedDict):
    slug: T
    details: str

class ContentDerivationPerspective(TypedDict):
    role: Description[AudienceRole]
    relationship_to_project: Description[AudienceToProjectRelationship]
    summary: str

ROLES = {
    "end-user": {
        "slug": "end-user",
        "details": "someone who uses the project we're working on to solve the problems or finish some tasks"
    },
    "maintainer": {
        "slug": "maintainer",
        "details": "someone who work with the project, but doesn't use it directly; someone that makes sure that "
                   "end-users can use it"
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
    ("end-user", "user"): "...",
    ("end-user", "developer"): "...",
    ("maintainer", "user"): "...",
    ("maintainer", "developer"): "...",
}

def perspective(role: AudienceRole, rel: AudienceToProjectRelationship) -> ContentDerivationPerspective:
    return {
        "role": ROLES[role],
        "relationship_to_project": LITERACY[rel],
        "summary": SUMMARY[(role, rel)]
    }

PERSPECTIVES = list(
    cross_product(
        {
            "role": expand_on(AudienceRole),
            "relationship_to_project": expand_on(AudienceToProjectRelationship)
        }
    )
)
from typing import Any

#todo document last param
def embed_metadata(data: dict[str, Any], prefix: str | None, fields: list[str] = [], strip_prefix: str | None = None) -> dict[str, Any]:
    """
    Small utility that comes in handy when copying some dict values to another one by possibly prefixing the key.
    Use as

        derived_doc = Document(
            ...,
            metadata={
                ...,
                **embed_metadata(derived_from.metadata, "foo", ["x", "y"])
            }
        )

    which adds to derived_doc.metadata keys "foo_x" and "foo_y", with values from data["x"] and data["y"] respectively.
    If key is missing, omits the copy. If no specific field list given, copies the whole data. If prefix is None, no
    prefix is applied.
    """
    if strip_prefix:
        def _strip_prefix(x: str) -> str:
            if x.startswith(strip_prefix):
                return x[len(strip_prefix):]
            return x
    else:
        _strip_prefix = lambda x: x
    return {
        _strip_prefix(prefix+"_"+k if prefix is not None else k): data[k]
        for k in ([f for f in fields if f in data] or data.keys())
    }
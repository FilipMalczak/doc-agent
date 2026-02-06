def to_simple_xml(data):
    """Convert a dict/list/primitive into simple XML text."""
    return "\n".join(_xml_lines(data))


def _xml_lines(data, key=None, indent=0):
    """
    Internal recursive generator.
    - data: any JSON-like value
    - key: optional XML tag name
    """
    pad = "  " * indent
    lines = []

    # ---- PRIMITIVES ---------------------------------------------------------
    if not isinstance(data, (dict, list)):
        if key is None:
            # top-level primitive: value only
            lines.append(str(data))
            return lines

        if isinstance(data, str) and "\n" in data:
            # multiline primitive
            lines.append(f"{pad}<{key}>")
            lines.append(data)
            lines.append(f"{pad}</{key}>")
        else:
            lines.append(f"{pad}<{key}>{data}</{key}>")
        return lines

    # ---- DICT ---------------------------------------------------------------
    if isinstance(data, dict):
        if key is None:
            # top-level dict: emit children only
            for k, v in data.items():
                lines.extend(_xml_lines(v, key=k, indent=indent))
            return lines

        # dict inside a tag
        lines.append(f"{pad}<{key}>")
        for k, v in data.items():
            lines.extend(_xml_lines(v, key=k, indent=indent + 1))
        lines.append(f"{pad}</{key}>")
        return lines

    # ---- LIST ---------------------------------------------------------------
    if isinstance(data, list):
        if key is None:
            lines.append(f"{pad}<list>")
            ind = 2
        else:
            lines.append(f"{pad}<{key}>")
            lines.append(f"{pad}  <list>")
            ind = 3
        for i, item in enumerate(data):
            item_pad = pad + "    "
            if isinstance(item, (dict, list)):
                # nested structure as separate <item>
                lines.append(f"{item_pad}<item index=\"{i}\">")
                lines.extend(_xml_lines(item, key=None, indent=indent + ind))
                lines.append(f"{item_pad}</item>")
            else:
                # primitive
                if isinstance(item, str) and "\n" in item:
                    lines.append(f"{item_pad}<item index=\"{i}\">")
                    lines.append(item)
                    lines.append(f"{item_pad}</item>")
                else:
                    lines.append(f"{item_pad}<item index=\"{i}\">{item}</item>")
        if key is None:
            lines.append(f"{pad}</list>")
        else:
            lines.append(f"{pad}  </list>")
            lines.append(f"{pad}</{key}>")
        return lines

    raise TypeError("Unsupported type")
def dict_to_simple_xml_lines(data, indent=0):
    """Convert a nested dict (JSON/YAML-like structure) to simple XML."""
    lines = []
    pad = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # nested dict
                lines.append(f"{pad}<{key}>")
                lines.extend(dict_to_simple_xml_lines(value, indent + 1))
                lines.append(f"{pad}</{key}>")

            elif isinstance(value, list):
                # list => <key><list>...</list></key>
                lines.append(f"{pad}<{key}>")
                lines.append(f"{pad}  <list>")
                for i, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        # nested structure inside item
                        lines.append(f"{pad}    <item index=\"{i}\">")
                        lines.extend(dict_to_simple_xml_lines(item, indent + 3))
                        lines.append(f"{pad}    </item>")
                    else:
                        # primitive item
                        if isinstance(item, str) and "\n" in item:
                            # multi-line
                            lines.append(f"{pad}    <item index=\"{i}\">")
                            lines.append(item)
                            lines.append(f"{pad}    </item>")
                        else:
                            lines.append(f"{pad}    <item index=\"{i}\">{item}</item>")
                lines.append(f"{pad}  </list>")
                lines.append(f"{pad}</{key}>")

            else:
                # primitive value
                if isinstance(value, str) and "\n" in value:
                    # multiline string
                    lines.append(f"{pad}<{key}>")
                    lines.append(value)  # no indentation applied inside
                    lines.append(f"{pad}</{key}>")
                else:
                    # simple inline value
                    lines.append(f"{pad}<{key}>{value}</{key}>")

    else:
        raise TypeError("Top-level structure must be a dict.")

    return lines


def to_simple_xml(data):
    return "\n".join(dict_to_simple_xml_lines(data))
from docassist.simple_xml import to_simple_xml


# ---------------------------------------------------------------------------
# PRIMITIVES
# ---------------------------------------------------------------------------

def test_top_level_string():
    assert to_simple_xml("hello") == "hello"


def test_top_level_int():
    assert to_simple_xml(42) == "42"


def test_top_level_float():
    assert to_simple_xml(3.14) == "3.14"


def test_top_level_boolean_true():
    assert to_simple_xml(True) == "True"


def test_top_level_boolean_false():
    assert to_simple_xml(False) == "False"


def test_simple_string_value():
    data = {"x": "abc"}
    expected = "<x>abc</x>"
    assert to_simple_xml(data) == expected


def test_simple_int_value():
    data = {"x": 123}
    expected = "<x>123</x>"
    assert to_simple_xml(data) == expected


def test_simple_float_value():
    data = {"x": 1.5}
    expected = "<x>1.5</x>"
    assert to_simple_xml(data) == expected


def test_simple_boolean_value():
    data = {"x": True}
    expected = "<x>True</x>"
    assert to_simple_xml(data) == expected


def test_multiline_string_value():
    data = {"x": {"y": "abc\ndef"}}
    expected = (
        "<x>\n"
        "  <y>\n"
        "abc\n"
        "def\n"
        "  </y>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


def test_multiline_string_as_list_item():
    data = {"x": ["a\nb"]}
    expected = (
        "<x>\n"
        "  <list>\n"
        "    <item index=\"0\">\n"
        "a\n"
        "b\n"
        "    </item>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


# ---------------------------------------------------------------------------
# DICTS
# ---------------------------------------------------------------------------

def test_flat_dict():
    data = {"a": 1, "b": 2}
    expected = (
        "<a>1</a>\n"
        "<b>2</b>"
    )
    assert to_simple_xml(data) == expected


def test_nested_dict():
    data = {"a": {"b": {"c": 1}}}
    expected = (
        "<a>\n"
        "  <b>\n"
        "    <c>1</c>\n"
        "  </b>\n"
        "</a>"
    )
    assert to_simple_xml(data) == expected


def test_nested_dict_with_float_and_bool():
    data = {"a": {"b": {"c": 1.25, "d": False}}}
    expected = (
        "<a>\n"
        "  <b>\n"
        "    <c>1.25</c>\n"
        "    <d>False</d>\n"
        "  </b>\n"
        "</a>"
    )
    assert to_simple_xml(data) == expected


def test_empty_dict_value():
    data = {"x": {}}
    expected = (
        "<x>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


# ---------------------------------------------------------------------------
# LISTS
# ---------------------------------------------------------------------------

def test_top_level_list_of_primitives():
    data = ["a", "b"]
    expected = (
        "<list>\n"
        "    <item index=\"0\">a</item>\n"
        "    <item index=\"1\">b</item>\n"
        "</list>"
    )
    assert to_simple_xml(data) == expected


def test_top_level_list_single_element():
    data = ["x"]
    expected = (
        "<list>\n"
        "    <item index=\"0\">x</item>\n"
        "</list>"
    )
    assert to_simple_xml(data) == expected


def test_top_level_list_with_float_and_bool():
    data = [1.0, False]
    expected = (
        "<list>\n"
        "    <item index=\"0\">1.0</item>\n"
        "    <item index=\"1\">False</item>\n"
        "</list>"
    )
    assert to_simple_xml(data) == expected


def test_nested_list_in_dict():
    data = {"x": ["a", "b"]}
    expected = (
        "<x>\n"
        "  <list>\n"
        "    <item index=\"0\">a</item>\n"
        "    <item index=\"1\">b</item>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


def test_list_of_dicts():
    data = {"x": [{"a": 1}, {"b": 2}]}
    expected = (
        "<x>\n"
        "  <list>\n"
        "    <item index=\"0\">\n"
        "      <a>1</a>\n"
        "    </item>\n"
        "    <item index=\"1\">\n"
        "      <b>2</b>\n"
        "    </item>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


def test_list_of_dicts_with_float_and_bool():
    data = {"x": [{"a": 1.1}, {"b": True}]}
    expected = (
        "<x>\n"
        "  <list>\n"
        "    <item index=\"0\">\n"
        "      <a>1.1</a>\n"
        "    </item>\n"
        "    <item index=\"1\">\n"
        "      <b>True</b>\n"
        "    </item>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


def test_list_of_lists():
    data = {"x": [[1, 2]]}
    expected = (
        "<x>\n"
        "  <list>\n"
        "    <item index=\"0\">\n"
        "      <list>\n"
        "          <item index=\"0\">1</item>\n"
        "          <item index=\"1\">2</item>\n"
        "      </list>\n"
        "    </item>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


def test_empty_list_value():
    data = {"x": []}
    expected = (
        "<x>\n"
        "  <list>\n"
        "  </list>\n"
        "</x>"
    )
    assert to_simple_xml(data) == expected


# ---------------------------------------------------------------------------
# MIXED / COMPLEX STRUCTURES
# ---------------------------------------------------------------------------

def test_complex_mixed_structure():
    data = {
        "a": 1,
        "b": {
            "c": ["x", {"y": "z"}],
            "d": "end",
        },
    }
    expected = (
        "<a>1</a>\n"
        "<b>\n"
        "  <c>\n"
        "    <list>\n"
        "      <item index=\"0\">x</item>\n"
        "      <item index=\"1\">\n"
        "        <y>z</y>\n"
        "      </item>\n"
        "    </list>\n"
        "  </c>\n"
        "  <d>end</d>\n"
        "</b>"
    )
    assert to_simple_xml(data) == expected


def test_complex_structure_with_float_and_bool():
    data = {
        "a": 1.0,
        "b": [
            True,
            {"c": False},
        ],
    }
    expected = (
        "<a>1.0</a>\n"
        "<b>\n"
        "  <list>\n"
        "    <item index=\"0\">True</item>\n"
        "    <item index=\"1\">\n"
        "      <c>False</c>\n"
        "    </item>\n"
        "  </list>\n"
        "</b>"
    )
    assert to_simple_xml(data) == expected


def test_deep_multiline_in_list_of_dicts():
    data = [{"x": "a\nb"}]
    expected = (
        "<list>\n"
        "    <item index=\"0\">\n"
        "    <x>\n"
        "a\n"
        "b\n"
        "    </x>\n"
        "    </item>\n"
        "</list>"
    )
    assert to_simple_xml(data) == expected


# ---------------------------------------------------------------------------
# STABILITY / DETERMINISM
# ---------------------------------------------------------------------------

def test_output_is_deterministic():
    data = {"x": ["a", "b", {"c": 1.0, "d": True}]}
    out1 = to_simple_xml(data)
    out2 = to_simple_xml(data)
    assert out1 == out2

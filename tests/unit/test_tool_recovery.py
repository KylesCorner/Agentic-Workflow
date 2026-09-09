from local_code_agent.agent.tool_recovery import recover_tool_calls


def test_recovers_qwen_tool_call():
    content = """
    <function=read_file>
    <parameter=path>src/main.py
    </tool_call>
    """

    calls = recover_tool_calls(content)

    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {
        "path": "src/main.py",
    }


def test_parses_json_values():
    content = """
    <function=search>
    <parameter=count>3
    <parameter=recursive>true
    <parameter=patterns>["*.py", "*.c"]
    </tool_call>
    """

    call = recover_tool_calls(content)[0]

    assert call.arguments["count"] == 3
    assert call.arguments["recursive"] is True
    assert call.arguments["patterns"] == ["*.py", "*.c"]


def test_multiple_tool_calls():
    content = """
    <function=read_file>
    <parameter=path>a.py
    </tool_call>

    <tool_call>
    <function=read_file>
    <parameter=path>b.py
    </tool_call>
    """

    calls = recover_tool_calls(content)

    assert [call.arguments["path"] for call in calls] == [
        "a.py",
        "b.py",
    ]


def test_normal_text_returns_no_calls():
    assert recover_tool_calls("Here is the answer.") == []

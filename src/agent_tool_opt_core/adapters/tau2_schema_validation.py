"""Construct TauBench tool schemas after candidate description edits.

The actual TauBench runtime owns schema generation. The adapter checks that
every tool still constructs, and reports changes to non-description schema
fields or lost documentation as diagnostics, matching the original adapter.
"""

from __future__ import annotations


def _tools(module):
    from tau2.environment.tool import Tool

    found = {}
    for name, value in vars(module).items():
        if getattr(value, "__module__", None) != module.__name__:
            continue
        if isinstance(value, type):
            instance = object.__new__(value)
            for method_name, method in vars(value).items():
                if getattr(method, "__tool__", False):
                    found[f"{name}.{method_name}"] = Tool(
                        getattr(instance, method_name)
                    )
        elif getattr(value, "__tool__", False):
            found[name] = Tool(value)
    return found


def validate_tool_schemas(module):
    return {
        name: {
            "short_description": tool.short_desc,
            "parameters": tool.openai_schema["function"]["parameters"],
            "returns": tool.returns.model_json_schema(),
        }
        for name, tool in _tools(module).items()
    }


def _structure(value):
    if isinstance(value, list):
        return [_structure(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _structure(item) for key, item in value.items() if key != "description"
    }


def validate_description_schemas(baseline, candidate):
    before = validate_tool_schemas(baseline)
    after = validate_tool_schemas(candidate)
    if before.keys() != after.keys():
        raise ValueError("tool set changed")
    diagnostics = []
    for name, original in before.items():
        edited = after[name]
        if (
            original["short_description"].strip()
            and not edited["short_description"].strip()
        ):
            diagnostics.append(f"{name}: lost tool description")
        for field in ("parameters", "returns"):
            previous = original[field]
            current = edited[field]
            if _structure(previous) != _structure(current):
                diagnostics.append(f"{name} {field}: non-description schema changed")
            for argument, schema in previous.get("properties", {}).items():
                if (
                    schema.get("description", "").strip()
                    and not current.get("properties", {})
                    .get(argument, {})
                    .get("description", "")
                    .strip()
                ):
                    diagnostics.append(
                        f"{name} {field}: lost description for {argument}"
                    )
    return diagnostics

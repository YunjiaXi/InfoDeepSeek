"""This module contains functions to fix JSON strings using general programmatic approaches, suitable for addressing
common JSON formatting issues."""
from __future__ import annotations

import contextlib
import json
import re
from typing import Optional

    
def find_json_list(input_str):
    try:
        # print(input_str)
        st = input_str.index("[")
        end = input_str.rindex("]")
        return input_str[st:end + 1]
    except:
        print('no list', input_str)
        return input_str


def find_json_dict(input_str, cnt=0):
    if input_str.count("{") > input_str.count("}"):
        return find_json_dict(input_str.rstrip("\n") + "\n}", cnt + 1)
    if cnt >= 5:
        return input_str
    try:
        st = input_str.index("{")
        end_str = '}\n}'
        end = input_str.rindex(end_str)
        return input_str[st:end + len(end_str)].strip()
    except json.decoder.JSONDecodeError:
        return find_json_dict(input_str.rstrip("\n") + "\n}", cnt + 1)
    except:
        return input_str


def extract_char_position(error_message: str) -> int:
    """Extract the character position from the JSONDecodeError message.

    Args:
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        int: The character position.
    """

    char_pattern = re.compile(r"\(char (\d+)\)")
    if match := char_pattern.search(error_message):
        return int(match[1])
    else:
        raise ValueError("Character position not found in the error message.")


def fix_invalid_escape(json_to_load: str, error_message: str) -> str:
    """Fix invalid escape sequences in JSON strings.

    Args:
        json_to_load (str): The JSON string.
        error_message (str): The error message from the JSONDecodeError
          exception.

    Returns:
        str: The JSON string with invalid escape sequences fixed.
    """
    while error_message.startswith("Invalid \\escape"):
        bad_escape_location = extract_char_position(error_message)
        json_to_load = (
            json_to_load[:bad_escape_location] + json_to_load[bad_escape_location + 1 :]
        )
        try:
            json.loads(json_to_load)
            return json_to_load
        except json.JSONDecodeError as e:
            print("json loads error - fix invalid escape", e)
            error_message = str(e)
    return json_to_load


def balance_braces(json_string: str) -> Optional[str]:
    """
    Balance the braces in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with braces balanced.
    """

    open_braces_count = json_string.count("{")
    close_braces_count = json_string.count("}")

    while open_braces_count > close_braces_count:
        json_string += "}"
        close_braces_count += 1

    while close_braces_count > open_braces_count:
        json_string = json_string.rstrip("}")
        close_braces_count -= 1

    with contextlib.suppress(json.JSONDecodeError):
        json.loads(json_string)
        return json_string


def add_quotes_to_property_names(json_string: str) -> str:
    """
    Add quotes to property names in a JSON string.

    Args:
        json_string (str): The JSON string.

    Returns:
        str: The JSON string with quotes added to property names.
    """

    def replace_func(match: re.Match) -> str:
        return f'"{match[1]}":'

    property_name_pattern = re.compile(r"(\w+):")
    corrected_json_string = property_name_pattern.sub(replace_func, json_string)

    try:
        json.loads(corrected_json_string)
        return corrected_json_string
    except json.JSONDecodeError as e:
        raise e


def correct_json(json_to_load: str):
    """
    Correct common JSON errors.
    Args:
        json_to_load (str): The JSON string.
    """

    try:
        json.loads(json_to_load)
        return json_to_load
    except json.JSONDecodeError as e:
        print("json loads error", e)
        error_message = str(e)
        if error_message.startswith("Invalid \\escape"):
            json_to_load = fix_invalid_escape(json_to_load, error_message)
        if error_message.startswith(
            "Expecting property name enclosed in double quotes"
        ):
            json_to_load = add_quotes_to_property_names(json_to_load)
            try:
                json.loads(json_to_load)
                return json_to_load
            except json.JSONDecodeError as e:
                print("json loads error - add quotes", e)
                error_message = str(e)
        if balanced_str := balance_braces(json_to_load):
            return balanced_str
    return json_to_load
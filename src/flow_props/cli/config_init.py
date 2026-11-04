"""Config initialization helpers for the flow-props CLI."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from flow_props.cli.common import ROOT_TEMPLATE

# --------------------------------------------------
# public API
# --------------------------------------------------


def write_full_config(config_path: Path, force: bool) -> None:
    """Write a full config file with all supported sections."""
    from flow_props.bl import TEMPLATE as BL_TEMPLATE
    from flow_props.entropy_layer import TEMPLATE as ENTROPY_TEMPLATE
    from flow_props.profiles import TEMPLATE as PROFILES_TEMPLATE
    from flow_props.wall import TEMPLATE as WALL_TEMPLATE

    # validate inputs
    if config_path.exists() and not force:
        typer.echo(
            f"error: config file already exists: {config_path}. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(code=1)

    # build the complete config text
    full_text = compose_full_config(
        ROOT_TEMPLATE,
        PROFILES_TEMPLATE,
        BL_TEMPLATE,
        ENTROPY_TEMPLATE,
        WALL_TEMPLATE,
    )

    # write the config file
    config_path.write_text(full_text)


def write_config_section(
    config_path: Path,
    section_name: str,
    section_text: str,
    force: bool,
) -> None:
    """Write one config section to a config file."""
    # read any existing config text from disk
    if config_path.exists():
        existing_text = config_path.read_text()
    else:
        existing_text = ""

    # ensure the root config keys exist exactly once
    config_text = ensure_root_keys(existing_text)

    # replace or append the requested section
    if has_section(config_text, section_name):
        if not force:
            typer.echo(
                f"error: [{section_name}] already exists in {config_path}. Use --force to overwrite.",
                err=True,
            )
            raise typer.Exit(code=1)

        config_text = replace_section(config_text, section_name, section_text)
    else:
        config_text = append_section(config_text, section_text)

    # write the updated config file
    config_path.write_text(config_text)


def ensure_section_exists(
    config_path: Path | None,
    section_name: str,
    section_text: str,
) -> bool:
    """Auto-append a missing section to an existing config file.

    If the config file exists but does not define the requested section, the
    template text for that section is appended in-place and the user is
    notified. Returns True when the file was modified.
    """
    # skip when we have no config file to update
    if config_path is None:
        return False
    if not config_path.is_file():
        return False

    # check whether the section already exists
    existing_text = config_path.read_text()
    if has_section(existing_text, section_name):
        return False

    # append the section template and notify the user
    updated_text = append_section(existing_text, section_text)
    config_path.write_text(updated_text)
    typer.echo(
        f"[info] [{section_name}] section was missing from {config_path}; "
        f"wrote default template. Edit it and re-run."
    )
    return True


def compose_full_config(*blocks: str) -> str:
    """Compose a config file from multiple TOML blocks."""
    # build clean text blocks with consistent spacing
    clean_blocks: list[str] = []
    for block in blocks:
        stripped_block = block.strip()
        if stripped_block:
            clean_blocks.append(stripped_block)

    return "\n\n".join(clean_blocks) + "\n"


def ensure_root_keys(config_text: str) -> str:
    """Ensure the root fname and gname keys exist."""
    # build the list of missing root keys
    missing_lines: list[str] = []
    if not has_root_key(config_text, "fname"):
        missing_lines.append('fname = "solution.vtu"         # path to CFD data file')
    if not has_root_key(config_text, "gname"):
        missing_lines.append('gname = ""                     # grid file for split formats only')

    # return early when nothing needs to be added
    if not missing_lines:
        if not config_text:
            return ROOT_TEMPLATE.strip() + "\n"
        return config_text if config_text.endswith("\n") else config_text + "\n"

    # insert missing root keys before the first section table
    lines = config_text.splitlines()
    insert_index = len(lines)
    for index, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith("[") and stripped_line.endswith("]"):
            insert_index = index
            break

    prefix_lines = lines[:insert_index]
    suffix_lines = lines[insert_index:]

    while prefix_lines and prefix_lines[-1] == "":
        prefix_lines.pop()

    updated_lines = prefix_lines + missing_lines
    if suffix_lines:
        updated_lines.append("")
        updated_lines.extend(suffix_lines)

    return "\n".join(updated_lines).rstrip() + "\n"


def has_root_key(config_text: str, key_name: str) -> bool:
    """Check whether a root config key exists."""
    pattern = rf"(?m)^\s*{re.escape(key_name)}\s*="
    return re.search(pattern, config_text) is not None


def has_section(config_text: str, section_name: str) -> bool:
    """Check whether a TOML section exists."""
    pattern = rf"(?m)^\[{re.escape(section_name)}\]\s*$"
    return re.search(pattern, config_text) is not None


def append_section(config_text: str, section_text: str) -> str:
    """Append a new section to the config text."""
    # build normalized config blocks before joining them
    existing_block = config_text.strip()
    new_block = section_text.strip()
    if not existing_block:
        return new_block + "\n"

    return existing_block + "\n\n" + new_block + "\n"


def replace_section(config_text: str, section_name: str, section_text: str) -> str:
    """Replace an existing section in the config text."""
    # locate the existing section boundaries
    lines = config_text.splitlines()
    start_index = None
    end_index = len(lines)
    header_line = f"[{section_name}]"

    for index, line in enumerate(lines):
        if line.strip() == header_line:
            start_index = index
            break

    if start_index is None:
        return append_section(config_text, section_text)

    for index in range(start_index + 1, len(lines)):
        stripped_line = lines[index].strip()
        if stripped_line.startswith("[") and stripped_line.endswith("]"):
            end_index = index
            break

    # replace the existing section text
    replacement_lines = section_text.strip().splitlines()
    updated_lines = lines[:start_index] + replacement_lines + lines[end_index:]
    return "\n".join(updated_lines).rstrip() + "\n"

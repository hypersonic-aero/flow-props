"""Pydantic config schema for flow-props."""

from __future__ import annotations

import builtins
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StationsConfig(BaseModel):
    """Station selection: no mode means all stations; otherwise choose one."""

    model_config = ConfigDict(extra="forbid")

    # -- mode 1: x-range --
    x_s: float | None = Field(default=None, description="Start wall-x [m]")
    x_e: float | None = Field(default=None, description="End wall-x [m]")
    dx: float | None = Field(default=None, gt=0, description="Step in x [m]")

    # -- mode 2: i-range --
    i_s: int | None = Field(default=None, description="Start i-index (inclusive)")
    i_e: int | None = Field(default=None, description="End i-index (inclusive)")
    di: int | None = Field(default=None, gt=0, description="Step in i")

    # -- mode 3: explicit list --
    list: builtins.list[int] | None = Field(default=None, description="Explicit i-station list")

    @model_validator(mode="after")
    def _check_exactly_one_mode(self) -> StationsConfig:
        """Validate that exactly one selection mode is fully specified."""
        # check which modes are (at least partially) filled
        mode1 = any(v is not None for v in [self.x_s, self.x_e, self.dx])
        mode2 = any(v is not None for v in [self.i_s, self.i_e, self.di])
        mode3 = self.list is not None
        active = sum([mode1, mode2, mode3])
        if active > 1:
            raise ValueError("stations: only one selection mode is allowed at a time")
        return self


class BLConfig(BaseModel):
    """Boundary-layer extraction options."""

    model_config = ConfigDict(extra="forbid")

    criterion: Literal["velocity_ratio", "htot_derivative"] = Field(
        default="velocity_ratio",
        description="BL edge detection criterion",
    )
    threshold: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Velocity ratio threshold (velocity_ratio criterion only)",
    )
    output: str = Field(
        default="bl_properties.dat",
        description="Output Tecplot ASCII file path",
    )
    profiles_output: str | None = Field(
        default=None,
        description="Optional Tecplot ASCII output for extracted wall-normal profiles",
    )


class GasConfig(BaseModel):
    """Optional gas-property overrides for calculations that require them."""

    model_config = ConfigDict(extra="forbid")

    gamma: float | None = Field(default=None, gt=1.0, description="Specific heat ratio")
    gas_constant: float | None = Field(default=None, gt=0.0, description="Gas constant [J/(kg K)]")


class FlowPropsConfig(BaseModel):
    """Top-level configuration for flow-props."""

    model_config = ConfigDict(extra="forbid")

    fname: str = Field(description="Path to CFD solution file")
    gname: str = Field(default="", description="Grid file for split-format CFD++")

    stations: StationsConfig = Field(default_factory=StationsConfig)
    bl: BLConfig = Field(default_factory=BLConfig)
    gas: GasConfig = Field(default_factory=GasConfig)


def load_config(config_path: Path) -> FlowPropsConfig:
    """Load and validate a flow-props TOML config file.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        Validated FlowPropsConfig instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValidationError: If the config fails validation.
    """
    # validate path
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    # read TOML
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # build and validate
    return FlowPropsConfig.model_validate(data)

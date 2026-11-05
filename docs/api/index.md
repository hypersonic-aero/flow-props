# API Reference

The public API is intentionally small and organized around the main post-processing tasks.

## Modules

- `flow_props.profiles` for wall-normal profile extraction
- `flow_props.bl` for boundary-layer edge detection and integral thicknesses
- `flow_props.entropy_layer` for entropy-layer thickness
- `flow_props.shock` for the shock point on each grid line and the shock curve
- `flow_props.wall` for wall shear and heat-transfer quantities
- `flow_props.cli` for command-line entry points

## Top-Level Imports

The package re-exports the most common functions through `flow_props` itself, so this style works:

```python
from flow_props import extract_profiles, extract_shock, run_bl, run_entropy, run_wall
```

Use the pages in this section for a module-by-module view.

# CLI

The command-line interface is exposed through `flow-props`. Every calculation is a
command group with the same commands:

```bash
flow-props <calculation> init       # write <calculation>.toml, every key documented
flow-props <calculation> run        # run <calculation>.toml
flow-props <calculation> examples   # bundled cases (calculations that ship some)
```

## Calculations

| Command | What it computes | Default config |
|---|---|---|
| `bl` | boundary-layer edge $\delta$, $\delta^*$, $\theta$, $H$ and edge state per station | `bl.toml` |
| `entropy-layer` | entropy-layer thickness per station | `entropy-layer.toml` |
| `profiles` | wall-normal profiles of every flow variable per station | `profiles.toml` |
| `shock` | the shock curve, and its offset from a shock-line grid's shock line | `shock.toml` |

## Global Options

Global options go before the calculation:

```bash
flow-props --version        # or -V
flow-props -v shock run     # --verbose: debug output from flow-props (not from cfd-io)
```

## init

```bash
flow-props bl init                    # writes bl.toml
flow-props bl init -o ogive_bl.toml   # another name (then: flow-props bl run -c ogive_bl.toml)
flow-props bl init --force            # overwrite an existing file
```

## run

`run` reads the calculation's default config, or the one named with `--config` / `-c`.
Paths in the config (solution file, outputs) are relative to the config file's folder.

```bash
flow-props bl run
flow-props shock run -c precursor.toml
```

Bad input (a misspelled key, a missing solution file, a missing field) is reported as one
`error: ...` line with exit code 1.

## examples

```bash
flow-props bl examples                       # list the bundled cases
flow-props bl examples sharp_flat_plate_m5   # copy its files here (as bl.toml)
flow-props bl run
```

An example is copied into the current folder; it refuses to overwrite existing files
unless `--force` is given.

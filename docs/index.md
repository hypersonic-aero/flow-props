# flow_props

`flow_props` extracts physically meaningful boundary-layer, entropy-layer and shock quantities from structured CFD datasets.

## Quick Start

### Install

```bash
pip install flow-props
```

### Run

=== "CLI"

	```bash
	flow-props bl init
	flow-props bl run
	```

=== "API"

	```python
	import cfd_io
	from flow_props import run_bl

	dataset = cfd_io.read_file("solution.vtu")
	results = run_bl(dataset, stations=[10, 50, 100])

	print(results[0])
	```

## Feedback & Contributing

Questions, bug reports, and contributions are welcome. If something unexpected
comes up while using this package, or there are ideas for improvement, opening
an issue or starting a discussion is the best first step.

Using a label when opening an issue helps prioritize and track requests:

- [Ask a question](https://github.com/hypersonic-aero/flow-props/issues/new?labels=question)
- [Report a bug](https://github.com/hypersonic-aero/flow-props/issues/new?labels=bug)
- [Suggest a feature](https://github.com/hypersonic-aero/flow-props/issues/new?labels=enhancement)

## License

GNU General Public License v3.0 or later. See
[LICENSE](https://github.com/hypersonic-aero/flow-props/blob/main/LICENSE) for the
complete license terms.


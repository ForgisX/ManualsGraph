# Technical Reference: Data Schema

ManualsGraph relies on a set of structured JSON databases to maintain consistency across the pipeline. These files are located in `config-manuals-structure/`.

## `oems.json`
Contains a canonical list of Manufacturers (OEMs).

```json
{
  "id": "fanuc",
  "name": "FANUC",
  "aliases": ["Fanuc Robotics", "Fanuc Ltd"],
  "description": "Industrial robots and CNC systems"
}
```

## `machines.json`
The central registry for all processed machines and their associated manuals.

| Field | Description |
| :--- | :--- |
| `id` | Unique identifier (usually slugified OEM + Model) |
| `oem_id` | Foreign key to `oems.json` |
| `factory_module` | The high-level category (e.g., `plc_control`, `drives_motors`) |
| `manuals` | A list of paths to sorted PDF files for this machine |

## `factory_modules.json`
Defines the standard categories for industrial equipment. Use these to tag new machines during classification.

- `plc_control`
- `drives_motors`
- `hmi_display`
- `primary_machine`
- `robots`
- `safety_systems`
- `conveyors_logistics`
- `packaging`
- `inspection`
- `power_supply`
- `cooling_ventilation`
- `sensors_actuators`
- `pnuematics_hydraulics`
- `other`

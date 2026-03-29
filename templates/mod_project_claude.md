# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this mod does

<!-- One paragraph: what gameplay this mod adds or changes, and why. -->

## MCP tools available

Four tools are available via the `ck3-modding-toolkit` MCP server.
Always search before writing any script — Paradox's syntax is highly specific.

| Tool | When to use |
|---|---|
| `search_wiki` | Understand a mechanic, look up available effects/triggers, check file format rules |
| `search_game_files` | Find a real vanilla example of an event, decision, trait, modifier, etc. |
| `get_section(parent_id)` | Read the full wiki section or game script block for a result from either search tool |
| `validate_mod` | Validate the mod with ck3-tiger and get a filtered list of errors and warnings |

`search_wiki` and `search_game_files` return focused 256-token excerpts — concise and never overflow context.
Each result includes a `parent_id`. Call `get_section(parent_id)` only when the excerpt isn't enough
and you need the complete block (e.g. all options of an event, full decision logic).

**Workflow for any new script element:**
1. `search_wiki` — understand the concept and required fields
2. `search_game_files` — find a vanilla example; use `get_section` if you need the full block
3. Write the script, staying consistent with the examples found
4. `validate_mod` — fix all ERRORs, re-validate until clean

`search_game_files` accepts an optional `category` filter. Common values:
`events`, `decisions`, `traits`, `modifiers`, `scripted_effects`, `scripted_triggers`,
`on_action`, `buildings`, `culture`, `religion`, `character_interactions`, `artifacts`, `gui`

## Mod file structure

```
mod/
├── descriptor.mod       # mod metadata (name, version, tags, dependencies)
├── <mod_name>.mod       # copy of descriptor.mod for launcher
├── common/
│   ├── decisions/       # .txt — player and AI decisions
│   ├── events/          # .txt — event scripts (use namespace = <mod_prefix>)
│   ├── traits/          # .txt — character traits
│   ├── modifiers/       # .txt — static modifiers
│   ├── scripted_effects/
│   ├── scripted_triggers/
│   └── on_action/       # .txt — on_action hooks
├── events/              # .txt — alternative events location
├── localization/
│   └── english/         # .yml — UTF-8 BOM, key: "Displayed text"
└── gfx/                 # icons, portraits, etc.
```

## Key conventions

- **Namespace**: every event file must declare `namespace = <mod_prefix>` at the top; event ids are `<prefix>.<number>` (e.g. `mymod.0001`)
- **Localisation keys**: must end with `.t` (title), `.desc` (description), `.a`/`.b`/... (option names)
- **Localisation files**: saved as UTF-8 **with BOM**, first line is `l_english:`
- **Load order**: filenames are loaded alphabetically; prefix with `00_` to load early, `zz_` to load last
- **Overwriting vanilla**: place a file at the same path — it replaces the entire vanilla file, so copy it first

## Validation

Use the `validate_mod` MCP tool after every meaningful change.
It runs [ck3-tiger](https://github.com/amtep/ck3-tiger) and returns only errors and warnings —
no progress noise, just actionable output.

```
validate_mod(mod_path="C:\\path\\to\\this\\mymod.mod")
```

`mod_path` must point to the **`.mod` file** (not the mod folder).

Fix all `ERROR` items and re-validate until clean. `WARNING` items are worth fixing but not blocking.

If ck3-tiger is not installed, the tool will say so — install it from the link above and ensure it is on PATH.
# Node UI acceptance

Every new node, and every change to a node's widgets, must work in all three views:

1. Classic canvas with HaDeS widgets.
2. Classic canvas with the Suite's standard ComfyUI widgets option enabled.
3. Nodes 2.0 / Vue.

Verify the actual frontend, not only Python schemas or mocked widgets. Check editable inputs, socket connections, execution, cached results, reports / previews, zoom and manual resizing. Reports must remain visible when changing styles, wrap or scroll within their node, and be read-only in both rendering systems. Keep native output widgets separate from decorative widgets hidden by the HaDeS style switch. Use `options.read_only` for Vue as well as the legacy element's `readOnly` property.

For workflow serialization changes, also verify save / reload. Record the tested frontend version, mode coverage, fixes and untested limits in `tools/RESULTS.md`. Add regression tests for discovered failures. Do not report all views as validated when any is untested.

Run UI smoke tests in an isolated instance when possible; preserve the user's active workflow and settings. Persistent test images and experiment evidence belong under `I:/HaDeS_MOSS`. Do not commit or push; the user handles those personally.

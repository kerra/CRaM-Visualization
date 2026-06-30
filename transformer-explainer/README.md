# Qwen + CrAM Explainer

This directory contains the Svelte visualization and Python trace/audience
server for the combined Qwen + CrAM demo.

Use the installation and operation guide in
[`QWEN_CRAM_SETUP.md`](QWEN_CRAM_SETUP.md).

The browser does not run another model or implement CrAM. The Python backend
loads Qwen, imports the official parent `utils/re_weighting.py` CrAM method, and
sends captured masks, attentions, activations, and predictions to the browser.

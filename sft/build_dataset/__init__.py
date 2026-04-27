"""SFT dataset preparation for the Agentic-MLLM project.

Two tool-use SFT datasets are produced from DeepEyesV2_SFT:

  * ``zoom_only``  — only ``image_zoom_in_tool`` is invoked
  * ``diverse``    — ``image_zoom_in_tool`` + ``code_interpreter`` +
                     ``search`` + ``image_search``

Both datasets share the same pure-zoom-in trajectories.
"""

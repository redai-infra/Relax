# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang runtime monkey-patches applied at engine start-up.

Each patch lives in its own module and is gated by an env flag checked in
``relax.backends.sglang.sglang_engine._launch_server_with_patches``.
"""

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import atexit
import importlib
import os
import signal
import sys
from pathlib import Path

import ray
import yaml
from ray import serve


# Optional telemetry hook: if the RELAX_TELEMETRY_HOOK env var names an
# importable module, import it here so it can install any patches it needs
# (e.g. a metrics-forwarding shim) before Controller is referenced below.
# Silent no-op when the env var is unset or the named module is unavailable.
_telemetry_hook = os.environ.get("RELAX_TELEMETRY_HOOK")
if _telemetry_hook:
    try:
        importlib.import_module(_telemetry_hook)
    except ImportError:
        pass

from relax.core.controller import Controller  # noqa: E402
from relax.utils.arguments import parse_args  # noqa: E402
from relax.utils.logging_utils import get_logger  # noqa: E402
from relax.utils.utils import post_process_env  # noqa: E402


cur_file_dir = Path(__file__).absolute().parent.parent.parent
logger = get_logger(__name__)

# Global reference so signal handlers / atexit can reach the controller.
_ctrl: Controller | None = None
_shutdown_done = False


def _graceful_shutdown(sig=None, frame=None):
    """Shut down SGLang engines and Ray on SIGTERM / SIGINT / atexit."""
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True

    sig_name = signal.Signals(sig).name if sig else "atexit"
    logger.info(f"Graceful shutdown triggered ({sig_name}) — cleaning up SGLang engines...")
    if _ctrl is not None:
        try:
            _ctrl.shutdown()
        except Exception as e:
            logger.warning(f"Controller shutdown error during {sig_name}: {e}")
    if ray.is_initialized():
        try:
            serve.shutdown()
            ray.shutdown()
            logger.info("Ray shutdown successfully")
        except Exception as e:
            logger.warning(f"Ray shutdown error during {sig_name}: {e}")
    if sig is not None:
        sys.exit(128 + sig)


def main(args):
    global _ctrl

    # Load runtime_env from config so we can both pass it to ray.init and
    # explicitly to the Serve deployment. Ensure it's available even if Ray
    # is already initialized.
    with open(os.path.join(cur_file_dir, "configs/env.yaml")) as file:
        runtime_env = yaml.safe_load(file)

    runtime_env = post_process_env(args, runtime_env)
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env=runtime_env)
        logger.info("Ray initialized successfully")
        try:
            serve.start(
                http_options={"host": "0.0.0.0", "port": "8000"},
                detached=True,
            )
        except RuntimeError:
            pass

    ctrl = Controller(args, runtime_env)
    _ctrl = ctrl

    # Register signal handlers so that `ray job stop` (SIGTERM) triggers cleanup.
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    atexit.register(_graceful_shutdown)

    try:
        ctrl.training_loop()
    except Exception as e:
        logger.exception(f"Training loop failed with error: {e}")
        _graceful_shutdown()
        os._exit(1)

    logger.info("Main func successfully")
    # Gracefully shut down SGLang engine processes before tearing down Ray Serve.
    _graceful_shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)

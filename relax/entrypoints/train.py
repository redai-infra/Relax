# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import atexit
import os
import signal
import sys
from pathlib import Path

import ray
import yaml
from ray import serve

from relax.core.controller import Controller
from relax.utils.arguments import parse_args
from relax.utils.logging_utils import get_logger
from relax.utils.utils import post_process_env


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
            serve_host = os.environ.get("RELAX_SERVE_HOST", "0.0.0.0")
            serve_port = int(os.environ.get("RELAX_SERVE_PORT", "8000"))
            serve.start(
                http_options={"host": serve_host, "port": serve_port},
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

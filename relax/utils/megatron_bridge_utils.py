from contextlib import contextmanager


try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


def _patch_progress_tracking_show_elapsed():
    """Replace ``MegatronModelBridge._with_progress_tracking`` to show elapsed
    time instead of remaining time.

    Upstream uses ``TimeRemainingColumn`` which counts down to 00:00 and erases
    itself the moment the bar finishes, so the final wall-clock cost of the
    conversion is invisible. We swap in ``TimeElapsedColumn`` so the rendered
    duration stays on screen after completion.
    """
    try:
        import torch
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    except ImportError:
        return

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not show_progress:
            yield from tasks
            return

        bridge_name = self.__class__.__name__
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not is_main_rank,
        ) as progress:
            task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)
            for task in tasks:
                yield task
                progress.update(task_id, advance=1)

    MegatronModelBridge._with_progress_tracking = _with_progress_tracking


_patch_progress_tracking_show_elapsed()


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")

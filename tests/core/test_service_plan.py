# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

from relax.core.service_plan import build_service_plan


def _config(**overrides) -> SimpleNamespace:
    values = {
        "loss_type": "policy_loss",
        "advantage_estimator": "grpo",
        "debug_rollout_only": False,
        "debug_train_only": False,
        "fully_async": False,
        "hybrid": False,
        "true_on_policy_mode": False,
        "colocate": True,
        "use_kl_loss": False,
        "kl_coef": 0.0,
        "genrm_model_path": None,
        "sft_predict_interval": None,
        "use_opd": False,
        "opd_type": "sglang",
        "teacher_hf_checkpoint": None,
        "opd_teacher_routes": None,
        "resource": {"actor": [1, 8], "rollout": [1, 8]},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_service_plan_matches_sync_and_hybrid_placement():
    sync_plan = build_service_plan(_config())
    hybrid_plan = build_service_plan(_config(hybrid=True, fully_async=True))

    assert sync_plan.roles == ("actor", "rollout")
    assert sync_plan.shares_actor_rollout_pg is True
    assert sync_plan.total_required_gpus == 8

    assert hybrid_plan.roles == ("actor", "rollout")
    assert hybrid_plan.shares_actor_rollout_pg is False
    assert hybrid_plan.total_required_gpus == 16


def test_service_plan_resolves_fully_async_conditional_roles():
    resources = {
        "actor": [1, 4],
        "rollout": [1, 4],
        "advantages": [1, 0],
        "actor_fwd": [1, 2],
    }
    plan = build_service_plan(_config(fully_async=True, colocate=False, resource=resources))

    assert plan.required_roles == ("actor", "rollout", "advantages", "actor_fwd")
    assert "reference" not in plan.roles
    assert plan.total_required_gpus == 10

    kl_plan = build_service_plan(
        _config(
            fully_async=True,
            colocate=False,
            use_kl_loss=True,
            resource=resources,
        )
    )
    assert "reference" in kl_plan.required_roles
    assert any(error.code == "missing_role" and error.role == "reference" for error in kl_plan.errors)


def test_service_plan_keeps_explicit_optional_reference():
    plan = build_service_plan(
        _config(
            fully_async=True,
            colocate=False,
            resource={
                "actor": [1, 4],
                "rollout": [1, 4],
                "advantages": [1, 0],
                "actor_fwd": [1, 2],
                "reference": [1, 2],
            },
        )
    )

    assert "reference" not in plan.required_roles
    assert "reference" in plan.roles
    assert not plan.errors


def test_service_plan_allows_only_cpu_roles_to_use_zero_gpus():
    plan = build_service_plan(
        _config(
            fully_async=True,
            colocate=False,
            resource={
                "actor": [1, 0],
                "rollout": [1, 4],
                "advantages": [1, 0],
                "actor_fwd": [1, 2],
            },
        )
    )

    assert any(error.code == "gpu_required" and error.role == "actor" for error in plan.errors)
    assert not any(error.code == "gpu_required" and error.role == "advantages" for error in plan.errors)


def test_service_plan_handles_sft_and_debug_modes():
    sft_plan = build_service_plan(
        _config(
            loss_type="sft",
            resource={"sft": [1, 0], "actor": [1, 8]},
        )
    )
    rollout_only_plan = build_service_plan(
        _config(
            debug_rollout_only=True,
            resource={"rollout": [1, 4]},
        )
    )

    assert sft_plan.roles == ("sft", "actor")
    assert sft_plan.total_required_gpus == 8
    assert rollout_only_plan.roles == ("rollout",)
    assert rollout_only_plan.total_required_gpus == 4


def test_service_plan_accounts_for_managed_teacher_placement():
    shared = build_service_plan(
        _config(
            use_opd=True,
            teacher_hf_checkpoint="/teacher",
            resource={"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]},
        )
    )
    dedicated = build_service_plan(
        _config(
            colocate=False,
            use_opd=True,
            teacher_hf_checkpoint="/teacher",
            resource={"actor": [1, 4], "rollout": [1, 4], "teacher": [1, 2]},
        )
    )

    shared_teacher = next(spec for spec in shared.planned_specs if spec.role == "teacher")
    dedicated_teacher = next(spec for spec in dedicated.planned_specs if spec.role == "teacher")
    assert shared_teacher.placement_group == "actor_rollout_shared"
    assert shared.total_required_gpus == 8
    assert dedicated_teacher.placement_group == "dedicated"
    assert dedicated.total_required_gpus == 10

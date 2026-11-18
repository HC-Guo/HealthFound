import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from verl.utils.hfpo_retention import validate_retention_config


@pytest.fixture
def launcher_environment(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    for command in ("ray", "sleep"):
        path = stub / command
        path.write_text("#!/bin/sh\nprintf '%s\\n' called >> \"$HFPO_TEST_RAY_CALLS\"\nexit 0\n")
        path.chmod(0o755)
    python = stub / "python3"
    python.write_text(
        "#!" + sys.executable + "\nimport json,os,sys\n"
        "if sys.argv[1:3] == ['-m','verl.trainer.main_ppo']:\n"
        "    open(os.environ['HFPO_TEST_ARGS'],'w').write(json.dumps(sys.argv[3:]))\n"
        "else:\n"
        "    os.execv(sys.executable,[sys.executable]+sys.argv[1:])\n"
    )
    python.chmod(0o755)
    env = dict(os.environ)
    for key in ("CONDA_ENV_NAME", "LOG_PATH", "RETENTION_LOSS_COEF", "ANCHOR_TRAIN_FILES_JSON"):
        env.pop(key, None)
    env.update(
        PATH=str(stub) + os.pathsep + env.get("PATH", ""),
        HFPO_TEST_ARGS=str(tmp_path / "args.json"), HFPO_TEST_RAY_CALLS=str(tmp_path / "ray_calls"),
        RANK="0", WORLD_SIZE="1", NNODES="1", GPUS_PER_NODE="2", NGPUS_PER_NODE="2",
        MODEL_PATH="/synthetic/model", TRAIN_FILES_JSON='["/synthetic/rl_train.parquet"]',
        VAL_FILES_JSON='["/synthetic/rl_dev.parquet"]', OUTPUT_DIR=str(tmp_path / "output"),
        PROJECT_NAME="synthetic_test", EXPERIMENT_NAME="synthetic_test",
    )
    return env


def launch(env):
    script = Path(__file__).resolve().parents[3] / "run_rl_grpo.sh"
    return subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, timeout=20)


def test_launcher_passes_valid_hfpo_overrides(launcher_environment):
    env = launcher_environment
    env.update(RETENTION_LOSS_COEF="0.2", ANCHOR_TRAIN_FILES_JSON='["/synthetic/anchors.jsonl"]', ANCHOR_BATCH_SIZE="4")
    result = launch(env)
    assert result.returncode == 0, result.stderr
    overrides = json.loads(Path(env["HFPO_TEST_ARGS"]).read_text())
    config_dir = Path(__file__).resolve().parents[2] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    assert validate_retention_config(config)
    assert config.actor_rollout_ref.actor.retention_loss_coef == .2
    assert config.data.retention.train_files == ["/synthetic/anchors.jsonl"]
    assert config.data.retention.batch_size == 4


@pytest.mark.parametrize("value", [None, "-0.1", "nan", "inf", "not-a-number"])
def test_launcher_rejects_missing_or_invalid_coefficient_before_ray(launcher_environment, value):
    env = launcher_environment
    if value is not None:
        env["RETENTION_LOSS_COEF"] = value
    result = launch(env)
    assert result.returncode != 0
    assert not Path(env["HFPO_TEST_RAY_CALLS"]).exists()


def test_launcher_positive_coefficient_requires_anchor_files(launcher_environment):
    env = launcher_environment
    env["RETENTION_LOSS_COEF"] = "0.1"
    result = launch(env)
    assert result.returncode != 0
    assert "ANCHOR_TRAIN_FILES_JSON" in result.stderr
    assert not Path(env["HFPO_TEST_RAY_CALLS"]).exists()


def test_launcher_can_explicitly_disable_retention(launcher_environment):
    env = launcher_environment
    env["RETENTION_LOSS_COEF"] = "0"
    result = launch(env)
    assert result.returncode == 0, result.stderr
    overrides = json.loads(Path(env["HFPO_TEST_ARGS"]).read_text())
    assert "actor_rollout_ref.actor.retention_loss_coef=0" in overrides
    assert "data.retention.train_files=[]" in overrides

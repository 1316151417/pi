"""`pi --self-test` proves both runtimes in one command."""

from __future__ import annotations

from pi_simple_cli.cli import run_self_test


async def test_cli_self_test_covers_the_full_runtime(capsys):
    assert await run_self_test() == 0
    output = capsys.readouterr().out
    assert "streaming, tool calling, transcript all OK" in output
    assert "harness tools (write/bash) executed on the real filesystem" in output
    assert "AgentHarness runtime ran a tool batch" in output

import subprocess


def run_cmd(cmd, capture_output=False):
    return subprocess.run(
        cmd,
        shell=True,
        check=True,
        executable="/bin/bash",
        capture_output=capture_output,
    )

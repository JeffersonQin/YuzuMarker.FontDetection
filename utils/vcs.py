from pathlib import Path

import pygit2


def get_current_tag() -> str:
    repo = pygit2.Repository(Path(__file__).parent.absolute())
    for file, val in repo.status().items():
        if val != 1 << 14:
            raise RuntimeError("Unstaged commit detected:", file, val)
    return repo.head.peel().short_id

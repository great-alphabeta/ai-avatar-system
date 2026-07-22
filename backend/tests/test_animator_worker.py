"""
Regression tests for the tracked MuseTalk worker shipped with the isolated
musetalk microservice (services/musetalk/musetalk_worker.py).
"""

from pathlib import Path


def test_tracked_worker_script_is_shipped():
    """The tracked worker must exist outside gitignored models/."""
    tracked = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "musetalk"
        / "musetalk_worker.py"
    )
    assert tracked.is_file(), (
        "services/musetalk/musetalk_worker.py is missing — MuseTalk service will not start"
    )


def test_animator_is_http_client():
    """AvatarAnimator talks to the musetalk microservice over HTTP."""
    from app.services.animator import AvatarAnimator

    animator = AvatarAnimator()
    assert animator.base_url
    assert hasattr(animator, "health")
    assert hasattr(animator, "animate")
    # Local worker spawning is gone — no _resolve_worker_script.
    assert not hasattr(animator, "_resolve_worker_script")

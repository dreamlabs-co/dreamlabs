"""Tests of `dream.models` against an in-process fake of the platform's `GET /v1/models`, over real HTTP."""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import PLATFORM_KEY as KEY
from conftest import FakePlatform

import dream
from dream import models
from dream.errors import DreamAuthError, DreamProtocolError

## The catalog the fake answers: a public base model, one of the account's fine-tunes, not yet servable, and a
## fine-tune of a custom body another account holds (a company running a body Dream Labs registered).
ARM_BODY = {
    "state": {"arm": {"start": 0, "end": 6}},
    "action": {"arm": {"start": 0, "end": 6}},
    "video": {"top": "observation.images.top"},
}
CATALOG = [
    {
        "id": "dreamlabs/dream-1-beta",
        "owner": "public",
        "base_model": "dreamlabs/dream-1-beta",
        "embodiment": "yam",
        "servable": True,
        "created_at": "2026-09-01T00:00:00Z",
        "cameras": ["overhead", "wrist_l", "wrist_r"],
        "embodiment_config": None,
    },
    {
        "id": "dream-1-ft:12345678-nb1",
        "owner": "mine",
        "base_model": "dreamlabs/dream-1-beta",
        "embodiment": "yam",
        "servable": False,
        "created_at": "2026-09-30T12:00:00Z",
        "cameras": ["overhead", "wrist_l", "wrist_r"],
        "embodiment_config": None,
    },
    {
        "id": "dream-1-ft:abcdef01-arm",
        "owner": "mine",
        "base_model": "dreamlabs/dream-1-beta",
        "embodiment": "emb_0123456789abcdef",
        "servable": True,
        "created_at": "2026-10-14T12:00:00Z",
        "cameras": ["top"],
        "embodiment_config": ARM_BODY,
    },
]


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakePlatform]:
    """The fake, named by `DREAM_API_URL` so every call reaches it; stopped after the test."""
    fake = FakePlatform(CATALOG)
    monkeypatch.setenv("DREAM_API_URL", fake.url)
    yield fake
    fake.close()


def test_list_models_returns_the_catalog(platform: FakePlatform) -> None:
    """list_models parses every row into a #Model, in the order the platform answered."""
    found = models.list_models(api_key=KEY)
    assert found == [
        models.Model(
            id="dreamlabs/dream-1-beta",
            owner="public",
            base_model="dreamlabs/dream-1-beta",
            embodiment="yam",
            servable=True,
            created_at=datetime.datetime(2026, 9, 1, tzinfo=datetime.timezone.utc),
            cameras=["overhead", "wrist_l", "wrist_r"],
        ),
        models.Model(
            id="dream-1-ft:12345678-nb1",
            owner="mine",
            base_model="dreamlabs/dream-1-beta",
            embodiment="yam",
            servable=False,
            created_at=datetime.datetime(2026, 9, 30, 12, 0, tzinfo=datetime.timezone.utc),
            cameras=["overhead", "wrist_l", "wrist_r"],
        ),
        models.Model(
            id="dream-1-ft:abcdef01-arm",
            owner="mine",
            base_model="dreamlabs/dream-1-beta",
            embodiment="emb_0123456789abcdef",
            servable=True,
            created_at=datetime.datetime(2026, 10, 14, 12, 0, tzinfo=datetime.timezone.utc),
            cameras=["top"],
            embodiment_config=ARM_BODY,
        ),
    ]


def test_list_models_names_this_client_and_its_version_as_the_user_agent(platform: FakePlatform) -> None:
    """The platform lists a body's parts only to a dream client that converts them, which it knows by this header."""
    models.list_models(api_key=KEY)
    assert platform.agents == [f"dream-client/{dream.__version__}"]


def test_a_model_whose_bodys_parts_are_withheld_says_so(platform: FakePlatform) -> None:
    """`parts_withheld` reads as the platform answers it; a platform from before the field lists False."""
    platform.catalog = [{**CATALOG[2], "parts_withheld": True}, CATALOG[0]]
    withheld, base = models.list_models(api_key=KEY)
    assert (withheld.parts_withheld, base.parts_withheld) == (True, False)


def test_a_model_listed_without_its_date_reads_none(platform: FakePlatform) -> None:
    """A Dream Labs model the platform lists without `created_at` (while it hides model dates) reads None, not a
    refusal; the account's own fine-tune keeps its date."""
    platform.catalog = [{k: v for k, v in CATALOG[0].items() if k != "created_at"}, CATALOG[1]]
    public, mine = models.list_models(api_key=KEY)
    assert (public.id, public.created_at) == ("dreamlabs/dream-1-beta", None)
    assert mine.created_at == datetime.datetime(2026, 9, 30, 12, 0, tzinfo=datetime.timezone.utc)


def test_a_bad_key_is_an_auth_error(platform: FakePlatform) -> None:
    """A key the platform rejects is `DreamAuthError`, the same as the data service."""
    with pytest.raises(DreamAuthError) as refused:
        models.list_models(api_key="dl_live_wrong")
    assert f"called {platform.url}" in str(refused.value), "the refusal names the URL it was sent to"
    assert "keys work only on the site that made them (dev, staging and production are separate)" in str(refused.value)


def test_host_defaults_to_the_platform_url_from_credentials(
    platform: FakePlatform, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key and the platform's URL both come from the credentials (`DREAM_API_URL`) when neither is passed."""
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "credentials"))
    monkeypatch.delenv("DREAM_API_KEY", raising=False)
    monkeypatch.delenv("DREAM_API_URL", raising=False)
    (tmp_path / "credentials").write_text(f"DREAM_API_KEY={KEY}\nDREAM_API_URL={platform.url}\n", encoding="utf-8")
    assert [model.id for model in models.list_models()] == [
        "dreamlabs/dream-1-beta",
        "dream-1-ft:12345678-nb1",
        "dream-1-ft:abcdef01-arm",
    ]


def test_a_platform_older_than_the_library_is_named_as_such(platform: FakePlatform) -> None:
    platform.catalog = [{key: value for key, value in CATALOG[0].items() if key != "cameras"}]
    with pytest.raises(DreamProtocolError, match="the platform is older than this library"):
        models.list_models(api_key=KEY)


def test_a_platform_from_before_body_configs_lists_its_models_with_none(platform: FakePlatform) -> None:
    """A platform that sends no `embodiment_config`: every model still lists, with None, so a run that looks its
    model up at start (every `ik` run) starts instead of being refused."""
    platform.catalog = [
        {key: value for key, value in entry.items() if key != "embodiment_config"} for entry in CATALOG
    ]
    listed = models.list_models(api_key=KEY)
    assert [model.embodiment_config for model in listed] == [None] * len(CATALOG)
    assert [model.cameras for model in listed] == [entry["cameras"] for entry in CATALOG]

"""Tests for the Core DPI catalogue manager."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unifi_core.network.managers.dpi_manager import DpiManager


@pytest.fixture
def connection():
    conn = MagicMock()
    conn.site = "default"
    conn.host = "192.168.1.1"
    conn.port = 443
    conn.verify_ssl = True
    conn.get_cached = MagicMock(return_value=None)
    conn._update_cache = MagicMock()
    return conn


@pytest.fixture
def manager(connection):
    auth = MagicMock()
    auth.has_api_key = True
    auth.get_api_key_session = AsyncMock()
    return DpiManager(connection, auth)


@pytest.mark.asyncio
async def test_full_catalogue_uses_supported_limit_and_actual_page_length(manager):
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_request(path, params=None):
        calls.append((path, params))
        if path == "/v1/dpi/categories":
            return {"data": [{"id": 4, "name": "Media streaming"}], "totalCount": 1}
        if params["offset"] == "0":
            return {
                "data": [{"id": 262274, "name": "Spotify"}, {"id": 262334, "name": "Netflix"}],
                "totalCount": 3,
            }
        if params["offset"] == "2":
            return {"data": [{"id": 262344, "name": "YouTube"}], "totalCount": 3}
        raise AssertionError(f"Unexpected request: {path} {params}")

    with patch.object(manager, "_request_integration_api", side_effect=fake_request):
        result = await manager.get_full_dpi_catalog()

    assert result == {
        "applications": [
            {"id": 262274, "name": "Spotify"},
            {"id": 262334, "name": "Netflix"},
            {"id": 262344, "name": "YouTube"},
        ],
        "categories": [{"id": 4, "name": "Media streaming"}],
    }
    assert calls == [
        ("/v1/dpi/applications", {"limit": "200", "offset": "0"}),
        ("/v1/dpi/applications", {"limit": "200", "offset": "2"}),
        ("/v1/dpi/categories", {"limit": "200", "offset": "0"}),
    ]


@pytest.mark.asyncio
async def test_full_catalogue_advances_by_each_returned_page_length(manager):
    offsets: list[str] = []

    async def fake_request(path, params=None):
        if path == "/v1/dpi/categories":
            return {"data": [], "totalCount": 0}
        offsets.append(params["offset"])
        pages = {
            "0": {"data": [{"id": 1}, {"id": 2}], "totalCount": 4},
            "2": {"data": [{"id": 3}], "totalCount": 4},
            "3": {"data": [{"id": 4}], "totalCount": 4},
        }
        return pages[params["offset"]]

    with patch.object(manager, "_request_integration_api", side_effect=fake_request):
        result = await manager.get_full_dpi_catalog()

    assert [entry["id"] for entry in result["applications"]] == [1, 2, 3, 4]
    assert offsets == ["0", "2", "3"]


@pytest.mark.asyncio
async def test_incomplete_later_page_is_not_cached(manager, connection):
    async def fake_request(path, params=None):
        if params["offset"] == "0":
            return {"data": [{"id": 262274, "name": "Spotify"}], "totalCount": 2}
        return None

    with (
        patch.object(manager, "_request_integration_api", side_effect=fake_request),
        pytest.raises(RuntimeError, match="incomplete DPI application catalogue"),
    ):
        await manager.get_full_dpi_catalog()

    connection._update_cache.assert_not_called()


@pytest.mark.asyncio
async def test_missing_api_key_fails_without_caching(connection):
    auth = MagicMock()
    auth.has_api_key = False
    manager = DpiManager(connection, auth)

    with pytest.raises(RuntimeError, match="failed to fetch DPI application catalogue"):
        await manager.get_full_dpi_catalog()

    connection._update_cache.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_page",
    [
        {"data": [{"id": 1}]},
        {"data": [{"id": 1}], "totalCount": 0},
        {"data": [{"id": 1}, {"id": 2}], "totalCount": 1},
    ],
)
async def test_invalid_first_page_total_count_is_not_cached(manager, connection, first_page):
    manager._request_integration_api = AsyncMock(return_value=first_page)

    with pytest.raises(RuntimeError, match="invalid totalCount"):
        await manager.get_full_dpi_catalog()

    connection._update_cache.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("later_page", "message"),
    [
        ({"data": [{"id": 2}, {"id": 3}], "totalCount": 4}, "changed totalCount"),
        ({"data": [{"id": 2}, {"id": 3}, {"id": 4}], "totalCount": 3}, "invalid totalCount"),
    ],
)
async def test_invalid_later_page_total_count_is_not_cached(manager, connection, later_page, message):
    manager._request_integration_api = AsyncMock(
        side_effect=[
            {"data": [{"id": 1}], "totalCount": 3},
            later_page,
        ]
    )

    with pytest.raises(RuntimeError, match=message):
        await manager._get_all_integration_pages("/v1/dpi/applications", "application")

    connection._update_cache.assert_not_called()


@pytest.mark.asyncio
async def test_integration_request_respects_connection_tls_policy(manager, connection):
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock(return_value={"data": []})
    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=response)
    context.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.get = MagicMock(return_value=context)
    session.close = AsyncMock()
    manager._auth.get_api_key_session.return_value = session

    await manager._request_integration_api("/v1/dpi/applications")

    assert session.get.call_args.kwargs["ssl"] is connection.verify_ssl

def test_monitor_settings_defaults(monkeypatch):
    for k in ("MONITOR_INTERVAL_SEC", "MONITOR_MAX_COMMENTS_PER_SCAN",
              "MONITOR_MAX_COMMENTS", "MONITOR_MAX_WATCHES", "MONITOR_STATE_PATH"):
        monkeypatch.delenv(k, raising=False)
    from app.backend.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    assert s.monitor_interval_sec == 300
    assert s.monitor_max_comments_per_scan == 30
    assert s.monitor_max_comments == 200
    assert s.monitor_max_watches == 20
    assert str(s.monitor_state_path).endswith("monitor_state.json")
    get_settings.cache_clear()


def test_monitor_settings_env_override(monkeypatch):
    monkeypatch.setenv("MONITOR_INTERVAL_SEC", "5")
    monkeypatch.setenv("MONITOR_MAX_WATCHES", "3")
    from app.backend.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    assert s.monitor_interval_sec == 5
    assert s.monitor_max_watches == 3
    get_settings.cache_clear()

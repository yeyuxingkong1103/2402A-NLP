def test_memories_route_registered():
    from app.main import create_app
    app = create_app()
    paths = app.openapi()["paths"]
    assert "/api/v1/sessions/{session_id}/memories" in paths
    assert "/api/v1/memories/{memory_id}" in paths
    assert "/api/v1/sessions/{session_id}/extract" in paths

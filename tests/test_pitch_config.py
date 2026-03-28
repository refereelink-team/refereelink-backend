from app.config.pitch import SoccerPitchConfiguration


def test_soccer_pitch_configuration_vertex_count() -> None:
    config = SoccerPitchConfiguration()

    assert len(config.vertices) == 32
    assert len(config.labels) == 32
    assert len(config.colors) == 32


def test_soccer_pitch_configuration_preserves_boundary_vertices() -> None:
    config = SoccerPitchConfiguration()

    assert config.vertices[0] == (0, 0)
    assert config.vertices[29] == (config.length, config.width)

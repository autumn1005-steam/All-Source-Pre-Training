import json
from pathlib import Path


def test_packages_and_provenance_are_present() -> None:
    import forward_pair
    import tempofound

    provenance = json.loads(Path("PROVENANCE.json").read_text(encoding="utf-8"))
    assert tempofound.__version__ == "1.0.0"
    assert forward_pair.__version__ == "1.0.0"
    assert (
        provenance["tempofound_commit"]
        == "560ca56e8b4ecdb63ad9ba82d1c16cf7c5e9a723"
    )
    assert (
        provenance["forward_pair_commit"]
        == "75357f19a0bb8f31aa8873b750d638181d0aec8a"
    )

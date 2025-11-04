def test_imports() :
    import importlib
    pkg = importlib.import_module("BtcSoloMinerGpu")
    assert hasattr(pkg , "__all__")



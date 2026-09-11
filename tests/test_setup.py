def test_core_libs_import():
    import pandas, sklearn, sentence_transformers, yaml, dotenv
    assert True

def test_pandas_works():
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert df["a"].sum() == 6

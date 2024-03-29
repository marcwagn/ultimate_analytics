from src.tasks import tasks
import pandas as pd
import numpy as np

def test_convert_to_final_results():
    clses = [0,  0,  0,  0,  0,  0, 34,  0, 36, 37,  0, 35,  0]
    xs = [21.736221,26.677188, 9.820513,22.107436,14.854319,13.040216, 0.000000,22.779594,18.684138,18.750717,12.455749,37.000000,14.495132]
    ys = [6.411272e+01,6.456938e+01,6.159928e+01,5.963575e+01,1.659033e+01,1.406071e+01,1.800000e+01,1.547850e+01,3.578895e+01,4.909965e+01,1.872962e+01,1.800000e+01,6.411298e+01]
    ids = np.random.randint(1, 100)
    teams = np.random.randint(0, 1)
    frames = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    data = dict(cls=clses, x=xs, y=ys, team=teams, id=ids, frame=frames)
    df = pd.DataFrame(data=data)

    final_results = tasks._convert_to_final_results(df)
    print(final_results)

    assert isinstance(final_results, dict)
    # Expecting 2 frames
    assert len(final_results.keys()) == 2
    assert "1" in final_results
    assert "2" in final_results

    assert isinstance(final_results["1"], list)
    assert isinstance(final_results["2"], list)

    expected_columns = ["cls", "x", "y", "team", "id"]
    assert set(final_results["1"][0].keys()).issuperset(set(expected_columns))
    assert final_results["1"][0]["x"] == 21.736221
    assert final_results["1"][0]["y"] == 6.411272e+01



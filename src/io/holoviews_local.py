from pathlib import Path

import mlflow 
import holoviews as hv
from kedro.io import AbstractVersionedDataSet


class HoloviewsWriter(AbstractVersionedDataSet):
    def __init__(self, fmt, filepath, version):
        super().__init__(Path(filepath), version)
        self._fmt = fmt

    def _load(self) -> pd.DataFrame:
        raise NotImplementedError("There is no way to convert from an arbitrary saved image format to a plot.")

    def _save(self, plot: hv.element.chart.Chart) -> None:
        save_path = self._get_save_path()
        hv.save(plot, save_path, fmt=self._fmt)

    def _describe(self):
        return dict(version=self._version, fmt=self._fmt)


class MLflowWriter(AbstractVersionedDataSet):
    def __init__(self, flavour, filepath, version):
        super().__init__(Path(filepath), version)
        self._flavour = flavour

    def _load(self, model_uri) -> pd.DataFrame:
        load_path = self._get_load_path()
        return getattr(mlflow, self._flavour).load_model(model_uri=load_path)

    def _save(self, model, path, conda_env=None, **kwargs) -> None:
        save_path = self._get_save_path()
        getattr(mlflow, self._flavour).save_model(model, path=save_path, conda_env=conda_env, **kwargs) 

    def _describe(self):
        return dict(version=self._flavour)

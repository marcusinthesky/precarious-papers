from pathlib import Path

import pandas as pd
import holoviews as hv
from kedro.io import AbstractVersionedDataSet


class HoloviewsWriter(AbstractVersionedDataSet):
    def __init__(self, filepath, version, fmt="png"):
        super().__init__(Path(filepath), version)
        self._fmt = fmt

    def _load(self) -> pd.DataFrame:
        load_path = self._get_load_path()

        if load_path.endswith(".png"):
            return hv.RGB.load_image(load_path)
        else:
            raise NotImplementedError(
                "There is no way to convert from an\
                                    arbitrary saved image format to a plot."
            )

    def _save(self, plot: hv.element.chart.Chart) -> None:
        save_path = self._get_save_path()
        hv.save(plot, save_path, fmt=self._fmt)

    def _describe(self):
        return dict(version=self._version, fmt=self._fmt)

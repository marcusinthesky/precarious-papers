import warnings
from pathlib import Path
from typing import Dict

from kedro.io import AbstractVersionedDataSet

from mlflow import pyfunc, sklearn
from mlflow.models import Model
from mlflow.utils.environment import _mlflow_conda_env
from precarious_papers.run import ProjectContext

CONTEXT = ProjectContext(str(Path.cwd()))


class _PythonModelWrapper(pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        pass

    def predict(self, context=None, model_input=None, X=None):
        if model_input is None:
            return self.model.predict(X)
        else:
            return self.model.predict(model_input)


class MLflowPyfuncDataset(AbstractVersionedDataSet):
    def __init__(
        self,
        filepath,
        version,
        log_model=True,
        conda_path=CONTEXT.project_path / "environment.yml",
        pip_path=CONTEXT.project_path / "requirements.txt",
        code_path=CONTEXT.project_path / "src" / "precarious_papers",
    ):
        super().__init__(Path(filepath), version)
        self._conda_path = Path(conda_path)
        self._log_model = log_model
        self._pip_path = Path(pip_path)
        self._code_path = Path(code_path)

    def _load(self, model_uri) -> Model:
        load_path = self._get_load_path()
        return pyfunc.load_model(model_uri=load_path)

    def _save(self, model, path, conda_env=None, **kwargs) -> None:
        save_path = self._get_save_path()

        _conda_env = _mlflow_conda_env(
            additional_pip_deps=(self._pip_path.open().read().split("\n")),
            additional_conda_deps=(self._conda_path.open().read().split("\n")),
        )

        if isinstance(model, pyfunc.PythonModel):
            if self._log_model:
                pyfunc.log_model(
                    python_model=model,
                    path=save_path,
                    conda_env=_conda_env,
                    code_path=self._code_path,
                    **kwargs
                )

            pyfunc.save_model(
                python_model=model,
                path=save_path,
                conda_env=_conda_env,
                code_path=self._code_path,
                **kwargs
            )
        else:
            warnings.warn("Not PythonModel, attempting to wrap model")
            _wrapped = _PythonModelWrapper(model)

            if self._log_model:
                pyfunc.log_model(
                    python_model=_wrapped,
                    path=save_path,
                    conda_env=_conda_env,
                    code_path=self._code_path,
                    **kwargs
                )
            pyfunc.save_model(
                python_model=_wrapped,
                path=save_path,
                conda_env=_conda_env,
                code_path=self._code_path,
                **kwargs
            )

    def _describe(self) -> Dict:
        return dict(version=self._version)


class MLflowSklearnDataset(AbstractVersionedDataSet):
    def __init__(
        self,
        filepath,
        version,
        log_model=True,
        conda_env=None,
        serialization_format="cloudpickle",
    ):
        super().__init__(Path(filepath), version)
        self._log_model = log_model
        self._conda_env = Path(conda_env)
        self._serialization_format = serialization_format

    def _load(self, model_uri) -> Model:
        load_path = self._get_load_path()
        return sklearn.load_model(model_uri=load_path)

    def _save(self, model, path, conda_env=None, **kwargs) -> None:
        save_path = self._get_save_path()

        if self._log_model:
            sklearn.log_model(
                sk_model=model,
                path=save_path,
                conda_env=self._conda_env,
                serialization_format=self._serialization_format,
                **kwargs
            )

        sklearn.save_model(
            sk_model=model,
            path=save_path,
            conda_env=self._conda_env,
            serialization_format=self._serialization_format,
            **kwargs
        )

    def _describe(self) -> Dict:
        return dict(version=self._version)

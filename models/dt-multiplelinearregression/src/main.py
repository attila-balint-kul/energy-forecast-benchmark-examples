from enfobench.dataset import Dataset
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge,LinearRegression
from enfobench.evaluation import evaluate_metrics
from enfobench.evaluation.metrics import mean_absolute_error, mean_bias_error,root_mean_squared_error
from enfobench import AuthorInfo, ModelInfo, ForecasterType
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration
from enfobench.dataset.utils import create_perfect_forecasts_from_covariates

class MultipleLinearRegressionDarts:

    def __init__(self, seasonality: str):
        self.seasonality = seasonality.upper()

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f"Darts.MultipleLinearRegression.{self.seasonality}",
            authors=[
                AuthorInfo(
                    name="Mohamad Khalil",
                    email="coo17619@newcastle.ac.uk"
                )
            ],
            
            type=ForecasterType.point,
            params={"seasonality": self.seasonality},
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame  | None = None,
        future_covariates: pd.DataFrame | None = None,
        **kwargs
        
    ) -> pd.DataFrame:
        
        periods = periods_in_duration(history.index, duration=self.seasonality)
        model=RegressionModel(lags= periods, output_chunk_length=horizon,model=LinearRegression())
        series = TimeSeries.from_dataframe(history, value_cols=['y'])
    
        #past_covariates = TimeSeries.from_dataframe(past_covariates, value_cols=['temperature'])
        #future_covariates = TimeSeries.from_dataframe(future_covariates, value_cols=['y'])
        #list(range(-336,0))
        model.fit(series,)
        
        # Make forecast
        pred = model.predict(horizon)

        forecast = (
            pred.pd_dataframe()
            .rename(columns={"y": "yhat"})
            .fillna(history['y'].mean())
        )
        
        return forecast
        
seasonality = os.getenv("ENFOBENCH_MODEL_SEASONALITY")
model = MultipleLinearRegressionDarts(seasonality)

app = server_factory(model)

    

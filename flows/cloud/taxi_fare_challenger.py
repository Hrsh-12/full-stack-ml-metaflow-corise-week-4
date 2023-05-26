from metaflow import FlowSpec, step, card, conda_base, current, Parameter, trigger, project, S3
from metaflow.cards import Markdown, Table, Image, Artifact

URL = 's3://outerbounds-datasets/taxi/latest.parquet'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


@trigger(events=['s3'])
@conda_base(libraries={'pandas': '1.4.2', 'pyarrow': '11.0.0', 'numpy': '1.21.2', 'scikit-learn': '1.1.2'})
@project(name="corise_week_4")
class TaxiFarePrediction(FlowSpec):

    data_url = Parameter("data_url", default=URL)

    def transform_features(self, df):
        # Remove noisy rows
        obviously_bad_data_filters = [
            df.fare_amount > 0,         # fare_amount in US Dollars
            df.trip_distance <= 100,    # trip_distance in miles
            df.trip_distance > 0,
            df.passenger_count > 0,
            df.total_amount > 0,
            df.trip_distance.notna(),
            df.total_amount.notna(),
        ]
        for f in obviously_bad_data_filters:
            df = df[f]
        
        return df

    @step
    def start(self):

        import pandas as pd
        from sklearn.model_selection import train_test_split


        with S3() as s3:
            obj = s3.get(URL)
            df = pd.read_parquet(obj.path)
        
        self.df = self.transform_features(df)
        self.X = self.df["trip_distance"].values.reshape(-1, 1)
        self.y = self.df["total_amount"].values
        self.next(self.linear_model)
    @step
    def linear_model(self):
        "Fit a single variable, linear model to the data."
        from sklearn.linear_model import LinearRegression

        # TODO: Play around with the model if you are feeling it.
        self.model = LinearRegression()

        self.next(self.validate)
    
    @card(type="corise")
    @step
    def validate(self):
        from sklearn.model_selection import cross_val_score
        self.scores = cross_val_score(self.model, self.X, self.y, cv=5)
        current.card.append(Markdown("# Taxi Fare Prediction Results"))
        current.card.append(Markdown(f"Score: {self.scores}"))
        self.next(self.end)

    @step
    def end(self):
        print("Success!")


if __name__ == "__main__":
    TaxiFarePrediction()

from flask import Blueprint, jsonify, request
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

form_routes = Blueprint('form_routes', __name__)

url = 'https://storage.googleapis.com/nutrikita-bucket/recipes_up_clean.csv'
dataset = pd.read_csv(url)
max_Calories = 2000
max_daily_fat = 100
max_daily_Saturatedfat = 13
max_daily_Cholesterol = 300
max_daily_Sodium = 2300
max_daily_Carbohydrate = 325
max_daily_Fiber = 40
max_daily_Sugar = 40
max_daily_Protein = 200
max_list = [max_Calories, max_daily_fat, max_daily_Saturatedfat, max_daily_Cholesterol, max_daily_Sodium,
            max_daily_Carbohydrate, max_daily_Fiber, max_daily_Sugar, max_daily_Protein]


@form_routes.route("/form", methods=["POST", "GET"])
def form_handler():
    if request.method == "POST":
        data = request.json
        result = data.values()

        # Convert object to a list
        data = list(result)

        # Convert list to an array
        numpyArray = np.array([data])

        def scaling(dataframe):
            scaler = StandardScaler()
            prep_data = scaler.fit_transform(dataframe.iloc[:, 16:25].to_numpy())
            return prep_data, scaler

        def nn_predictor(prep_data):
            neigh = NearestNeighbors(metric='cosine', algorithm='brute')
            neigh.fit(prep_data)
            return neigh

        def build_pipeline(neigh, scaler, params):
            transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
            pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
            return pipeline

        def extract_data(dataframe, ingredient_filter, max_nutritional_values):
            extracted_data = dataframe.copy()
            for column, maximum in zip(extracted_data.columns[16:25], max_nutritional_values):
                extracted_data = extracted_data[extracted_data[column] < maximum]
            if ingredient_filter is not None:
                for ingredient in ingredient_filter:
                    extracted_data = extracted_data[
                        extracted_data['RecipeIngredientParts'].str.contains(ingredient, regex=False)]
            return extracted_data

        def apply_pipeline(pipeline, _input, extracted_data):
            return extracted_data.iloc[pipeline.transform(_input)[0]]

        def recommend(dataframe, _input, max_nutritional_values, ingredient_filter=None, params={'return_distance': False}):
            extracted_data = extract_data(dataframe, ingredient_filter, max_nutritional_values)
            prep_data, scaler = scaling(extracted_data)
            neigh = nn_predictor(prep_data)
            pipeline = build_pipeline(neigh, scaler, params)
            return apply_pipeline(pipeline, _input, extracted_data)

        output = recommend(dataset, numpyArray, max_list)
        id_column = 'RecipeId'

        # Convert DataFrame to JSON format with 'records' orientation
        output_json = output.to_dict(orient='records')

        # Remove backslashes and convert specific properties to arrays
        for item in output_json:
            for key, value in item.items():
                if isinstance(value, str) and value != "NaN":
                    item[key] = value.replace('\\', '').replace('\"', '')

                # Convert specific properties to arrays
                if key in ['Keywords', 'RecipeIngredientParts', 'RecipeIngredientQuantities']:
                    if value and value != "NaN":
                        item[key] = [x.strip().strip('"') for x in value.split(',')]

        return jsonify(output_json)

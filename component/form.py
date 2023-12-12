from flask import Blueprint, jsonify, request
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


form_routes = Blueprint('form_routes', __name__)

url = './recipes.csv'
dataset = pd.read_csv(url)
max_Calories=2000
max_daily_fat=100
max_daily_Saturatedfat=13
max_daily_Cholesterol=300
max_daily_Sodium=2300
max_daily_Carbohydrate=325
max_daily_Fiber=40
max_daily_Sugar=40
max_daily_Protein=200
max_list=[max_Calories,max_daily_fat,max_daily_Saturatedfat,max_daily_Cholesterol,max_daily_Sodium,max_daily_Carbohydrate,max_daily_Fiber,max_daily_Sugar,max_daily_Protein]


@form_routes.route("/form", methods=["POST"])
def form_prediction():
    if request.method == "POST":
        data=request.json
        result = data.values()

        # Convert object to a list
        data = list(result)
        
        # Convert list to an array
        numpyArray = np.array([data])
        print(data)

    def scaling(dataframe):
        scaler=StandardScaler()
        prep_data=scaler.fit_transform(dataframe.iloc[:,16:25].to_numpy())
        return prep_data,scaler

    def nn_predictor(prep_data):
        neigh = NearestNeighbors(metric='cosine',algorithm='brute')
        neigh.fit(prep_data)
        return neigh

    def build_pipeline(neigh,scaler,params):
        transformer = FunctionTransformer(neigh.kneighbors,kw_args=params)
        pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
        return pipeline

    def extract_data(dataframe,ingredient_filter,max_nutritional_values):
        extracted_data=dataframe.copy()
        for column,maximum in zip(extracted_data.columns[16:25],max_nutritional_values):
            extracted_data=extracted_data[extracted_data[column]<maximum]
        if ingredient_filter!=None:
            for ingredient in ingredient_filter:
                extracted_data=extracted_data[extracted_data['RecipeIngredientParts'].str.contains(ingredient,regex=False)] 
        return extracted_data

    def apply_pipeline(pipeline,_input,extracted_data):
        return extracted_data.iloc[pipeline.transform(_input)[0]]

    def recommand(dataframe,_input,max_nutritional_values,ingredient_filter=None,params={'return_distance':False}):
        extracted_data=extract_data(dataframe,ingredient_filter,max_nutritional_values)
        prep_data,scaler=scaling(extracted_data)
        neigh=nn_predictor(prep_data)
        pipeline=build_pipeline(neigh,scaler,params)
        return apply_pipeline(pipeline,_input,extracted_data)
    
    output=recommand(dataset,numpyArray,max_list)
    id_column = 'RecipeId'
    
    # Convert DataFrame to JSON format with 'records' orientation
    output_json = output.to_dict(orient='records')
    
    # Create a new dictionary with modified keys
    modified_output = {str(entry[id_column]): entry for entry in output_json}
    
    # Save the modified JSON data to a file
    with open('output.json', 'w') as file:
        file.write(json.dumps(modified_output, indent=2))

    return "Received and processed the data successfully!" 

@form_routes.route("/form_results", methods=["GET"])
def get_results():
    try:
        # Read the data from the 'output.json' file
        with open('output.json', 'r') as file:
            data = json.load(file)
        
        # Make structur json file
        results = [
            {
                "RecipeId": entry["RecipeId"],
                "Name": entry["Name"],
                "Author": {
                    "AuthorId": entry.get("AuthorId", None),
                    "AuthorName": entry.get("AuthorName", None)
                },
                "CookTime": entry.get("CookTime", None),
                "PrepTime": entry.get("PrepTime", None),
                "TotalTime": entry.get("TotalTime", None),
                "DatePublished": entry.get("DatePublished", None),
                "Description": entry.get("Description", None),
                "Images": entry.get("Images", None),
                "RecipeCategory": entry.get("RecipeCategory", None),
                "Keywords": [f'c("{keyword.strip()}")' for keyword in str(entry.get("Keywords", "")).split(',') if entry.get("Keywords")],
                "RecipeIngredientParts": list({"c(\"" + entry_part.strip().lstrip('c("').rstrip('""') + '")' for entry_part in entry.get("RecipeIngredientParts", "").split(',') if entry.get("RecipeIngredientParts")}),
                "RecipeIngredientQuantities": [f'c("{quantity.strip()}")' for quantity in entry.get("RecipeIngredientQuantities", "").split(',') if entry.get("RecipeIngredientQuantities")],
                "AggregatedRating": entry.get("AggregatedRating", None),
                "ReviewCount": entry.get("ReviewCount", None),
                "Calories": entry.get("Calories", None),
                "FatContent": entry.get("FatContent", None),
                "SaturatedFatContent": entry.get("SaturatedFatContent", None),
                "CholesterolContent": entry.get("CholesterolContent", None),
                "SodiumContent": entry.get("SodiumContent", None),
                "CarbohydrateContent": entry.get("CarbohydrateContent", None),
                "FiberContent": entry.get("FiberContent", None),
                "SugarContent": entry.get("SugarContent", None),
                "ProteinContent": entry.get("ProteinContent", None),
                "RecipeServings": entry.get("RecipeServings", None),
                "RecipeYield": entry.get("RecipeYield", None),
                "RecipeInstructions": [f'c("{instruction.strip()}")' for instruction in entry.get("RecipeInstructions", "").split(',') if entry.get("RecipeInstructions")],
            }
            for entry in data.values()
        ]

        return jsonify(results)
    except FileNotFoundError:
        return jsonify({"error": "Output data not found"}), 404


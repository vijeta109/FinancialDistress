from typing import Optional
from urllib import request

import uvicorn
from fastapi import FastAPI,Response,Request
from fastapi.templating import Jinja2Templates
import json
from fastapi.templating import Jinja2Templates
from prediction_Validation_Insertion import pred_validation # need to fix
from trainmodel import trainModel
from training_Validation_Insertion import train_validation
#  import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
from typing import Dict
import pandas as pd



# For Cross Origin
app = FastAPI(debug = True)


templates = Jinja2Templates(directory="templates")

#route to dashboard(UI)
@app.get("/")
def dashboard(request: Request):
    """
    displays the financial distress prediction dashboard/homepage
    """
    return templates.TemplateResponse("dashboard.html", {
        "request": request
    })


@app.post("/predictclient")
async def predictRouteClient(json_data: Dict):
    try:
        print("Start Predicting")

        if json_data.get('Data') is not None:
            Data = json_data.get('Data')
            Df = pd.DataFrame.from_dict(Data)
            Df = Df.astype(float)
            Df = Df.T
            schema_path = 'schema_prediction_ui.json'
            with open(schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            column_names = dic['ColName']
            Df.columns = column_names
            pred_val = pred_validation(Df,'UI')  # object initialization

            pred_val.prediction_validation()  # calling the prediction_validation function

            pred = prediction(Df,'UI')  # object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % Data)

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)


@app.post("/predict")
async def predictRouteClient(json_data: Dict):

     try:
         print ("Start Predicting")
         if json_data.get('folderPath') is not None:
             path = json_data.get('folderPath')

             pred_val = pred_validation(path,'Batch') #object initialization

             pred_val.prediction_validation() #calling the prediction_validation function

             pred = prediction(path,'Batch') #object initialization

             # predicting for dataset present in database
             path = pred.predictionFromModel()
             return Response("Prediction File created at %s!!!" % path)

     except ValueError:
         return Response("Error Occurred! %s" %ValueError)
     except KeyError:
         return Response("Error Occurred! %s" %KeyError)
     except Exception as e:
         return Response("Error Occurred! %s" %e)


@app.post("/train")
async def trainRouteClient(json_data: Dict):

    
    
    try:
        if json_data.get('folderPath') is not None:
            path = json_data.get('folderPath')
            
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function


            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table
            return Response("Training is Completed")


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return 'Error Occurred!'
    #return Response("Training successfull!!")



if __name__ == "__main__" :
    uvicorn.run(app,host = "0.0.0.0",port = 8000)

# uvicorn app:app --port 5000
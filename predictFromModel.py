import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pandas as pd


class prediction:

    def __init__(self,path,Batch):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.Batch = Batch
        if self.Batch == 'Batch':
            self.pred_data_val = Prediction_Data_validation(path,'Batch')
        else:
            self.df = path

    def predictionFromModel(self):

        try:
            if self.Batch == 'Batch':
                self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
                self.log_writer.log(self.file_object,'Start of Prediction')
                data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
                data=data_getter.get_data()

                #code change

                preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

                # replacing '?' values with np.nan as discussed in the EDA part

                # check if missing values are present in the dataset
                is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
                if (is_null_present):
                    data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation


                # get encoded values for categorical data


                #data=data.to_numpy()
                file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
                kmeans=file_loader.load_model('KMeans')

                ##Code changed

                clusters=kmeans.predict(data.drop(labels=['class'], axis=1))#drops the first column for cluster prediction
                data['clusters']=clusters
                clusters=data['clusters'].unique()


                for i in clusters:
                    cluster_data = data[data['clusters'] == i]
                    class_names = list(cluster_data['class'])
                    cluster_data = data.drop(labels=['class'], axis=1)
                    cluster_data = cluster_data.drop(['clusters'], axis=1)
                    model_name = file_loader.find_correct_model_file(i)
                    model = file_loader.load_model(model_name)
                    result = list(model.predict(cluster_data))
                    result = pandas.DataFrame(list(zip(class_names, result)), columns=['class', 'Prediction'])
                    path = "Prediction_Output_File/Predictions.csv"
                    result.to_csv("Prediction_Output_File/Predictions.csv", header=True,
                                  mode='a+')  # appends result to predictio

                self.log_writer.log(self.file_object,'End of Prediction')
            else:
                preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
                is_null_present, cols_with_missing_values = preprocessor.is_null_present(self.df)
                if (is_null_present):
                    data = preprocessor.impute_missing_values(self.df,
                                                              cols_with_missing_values)  # missing value imputation
                else:
                    data = self.df

                # data=data.to_numpy()
                file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
                kmeans = file_loader.load_model('KMeans')

                ##Code changed

                clusters = kmeans.predict(data)  # drops the first column for cluster prediction
                data['clusters'] = clusters
                clusters = data['clusters'].unique()
                for i in clusters:
                    cluster_data = data[data['clusters'] == i]
                    cluster_data = cluster_data.drop(['clusters'], axis=1)
                    model_name = file_loader.find_correct_model_file(i)
                    model = file_loader.load_model(model_name)
                    result = pd.DataFrame(model.predict(cluster_data))
                    result.to_csv("Prediction_Output_File/Predictions.csv", header=True,
                                  mode='a+')  # appends result to predictio

                self.log_writer.log(self.file_object, 'End of Prediction')

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path






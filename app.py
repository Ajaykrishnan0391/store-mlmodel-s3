from flask import Flask, jsonify
import pandas as pd
from sklearn import linear_model
import pickle
from io import BytesIO
import boto3


app = Flask(__name__)
ma = Marshmallow(app)


@app.route('/')
def home():
    return "<p>Hello World</p>"


@app.route('/train-save-model-s3/', methods=['POST'])
def train_save_model_s3():    
    data = [[1500, 5700], [1800, 5500], [1900, 5450]]
    df = pd.DataFrame(data, columns=['area', 'price'])
    model = linear_model.LinearRegression()
    model.fit(df[['area']], df.price)
    save_model_s3(model)
    response = jsonify("Model trained successfully")
    return response

@app.route('/predict-model-s3/<input_val>', methods=['GET'])
def predict_model_s3(input_val):       
    model = get_model_s3()
    prediction = model.predict([[input_val]])
    response = jsonify({'predicted_cost': prediction[0]})
    response.status_code = 200
    return response


def save_model_s3(model):
    bucket = 'test'    
    s3_file_name = 'Test_model.pickle'
    s3_client = get_connection()
    try:
        with BytesIO() as f:
            pickle.dump(model, f)
            f.seek(0)
            s3_client.upload_fileobj(Bucket=bucket, Key=s3_file_name, Fileobj=f)
    except Exception as e:
        print(e)
    print("Upload Successful")   

def get_model_s3():
    bucket = 'test'    
    s3_file_name = 'Test_model.pickle'
    s3_client = get_connection()
    try:
        with BytesIO() as f:
            s3_client.download_fileobj(Bucket=bucket, Key=s3_file_name, Fileobj=f)
            f.seek(0)
            model = pickle.load(f)
            return model
    except Exception as e:
        print(e) 

def get_connection() -> boto3.client:
    end_point = '<s3 end point>'
    access_key_id = 'access_key_id'
    access_key = 'access_key'    
    cert_path = 'certificate path if required'
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=access_key,
                        endpoint_url=end_point, verify=cert_path)
    return s3        


@app.teardown_appcontext
def shutdown_session(exception=None):
    #db_session.remove()
    pass

if __name__ == '__main__':
    app.run(debug=True)

import os
import uuid
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, Blueprint, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "20222022"
app.config['MODEL_FOLDER'] = './models'
app.config['UPLOAD_FOLDER'] = './upload'
app.config['ALLOWED_EXTENSIONS'] = ('.nii.gz', '.dcm')
app.config['MAX_CONTENT_LENGTH'] = 1024**4
blueprint = Blueprint('api', __name__)
api = Api(blueprint, version='1.0', title='Prediction API',
    description='Predict segmentation for target')
app.register_blueprint(blueprint, url_prefix="/api")

upload_parser = api.parser()
upload_parser.add_argument('data', required=True, location='files',
    type=FileStorage)

predict_parser = api.parser()
predict_parser.add_argument('model', required=True, location='args',
    choices=['lung lobes', 'lungs covid', 'cancer'])
predict_parser.add_argument('data', required=True, location='args')


def check_extention(filename):
    is_supported = False
    for extention in app.config['ALLOWED_EXTENSIONS']:
        if filename.endswith(extention):
            is_supported = True
            break
    return is_supported


@api.route('/upload', doc={'description': 'Upload DICOM or NIfTI files'})
class Upload(Resource):
    @api.expect(upload_parser)
    @api.doc(params={
        'data': 'DICOM or NIfTI files',
    })
    @api.marshal_with(api.model('Upload', {
        'message': fields.String,
        'uuid': fields.String,
    }), code=201)
    def post(self):
        args = upload_parser.parse_args()
        data = args['data']
        if data.filename == '':
            return {
                'message' : 'No file selected for uploading',
            }, 400
        if not data or not check_extention(data.filename):
            return {
                'message' : f"Allowed file types are {app.config['ALLOWED_EXTENSIONS']}",
            }, 400
        data_uuid = str(uuid.uuid4())
        save_folder = os.path.join(app.config['UPLOAD_FOLDER'], data_uuid)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, secure_filename(data.filename))
        data.save(save_path)
        return {
            'message' : 'Successfully uploaded',
            'uuid': data_uuid,
        }, 201


@api.route('/predict', doc={'description': 'Load model and predict'})
class Predict(Resource):
    @api.expect(predict_parser)
    @api.doc(params={
        'data': 'Upload uuid',
        'model': 'Model target',
    })
    @api.marshal_with(api.model('Predict', {
        'message': fields.String,
        'uuid': fields.String,
    }), code=201)
    def post(self):
        args = predict_parser.parse_args()
        model = args['model']
        data = args['data']
        try:
            model = ''
            prediction = ''
        except Exception as e:
            return {
                'message' : f"Model error {e}",
            }, 500
        predict_uuid = str(uuid.uuid4())
        save_folder = os.path.join(app.config['PREDICT_FOLDER'], predict_uuid)
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, secure_filename(data.filename))
        # prediction save
        return {
            'message' : 'Successfully predicted',
            'uuid': predict_uuid,
        }, 201


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='80', debug=True)

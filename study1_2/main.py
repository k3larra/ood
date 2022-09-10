import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import os
import sys
import urllib
import json
import requests
from flask import Flask, flash, request, redirect, url_for, send_from_directory, after_this_request, jsonify, make_response
from werkzeug.utils import secure_filename
from io import StringIO ##For images
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from torchvision import models
from torchvision import transforms
from captum.attr import IntegratedGradients
#from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution
from captum.attr import GradientShap
from captum.attr import visualization as viz
#
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

#Do the torch thing
#Do the torch thing
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True #Skum grej från https://github.com/pytorch/pytorch/issues/61755
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()
if torch.cuda.is_available():
    print("Cuda:",file=sys.stderr)
    model.to('cuda')
if not torch.cuda.is_available():
    print("No Cuda:",file=sys.stderr) 

# model expects 224x224 3-color image
transform = transforms.Compose([
 transforms.Resize(224),
 transforms.CenterCrop(224),
 transforms.ToTensor()
])

preprocess = transforms.Compose([   #Used for predictions
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#For the vizualisation
method="blended_heat_map"
sign="positive"
alpha_overlay = 0.6
default_cmap = LinearSegmentedColormap.from_list('custom green', 
                                                 [(0, '#39422c'),
                                                  (1, '#8df505')], N=5)

# Download ImageNet labels
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename) 
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return app.send_static_file('index.html')

##Only teststuff from here to#########
@app.route('/hello', methods=['GET'])
def hello():
    @after_this_request
    def add_header(response):
        #response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    jsonResp={"jack":4098,"sape":4139}
    print(jsonResp)
    return jsonify(jsonResp)

@app.route('/test', methods=['GET', 'POST'])
def testfn():
    # GET request
    if request.method == 'GET':
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers
    # POST request
    if request.method == 'POST':
        print(request.get_json())  # parse as JSON
        return 'Sucesss', 200

######## Data fetch ############
@app.route('/getdata/<index_no>', methods=['GET','POST'])
def data_get(index_no):
    if request.method == 'POST': # POST request
        print(request.get_text())  # parse as text
        return 'OK', 200
    else: # GET request
        return 't_in = %s ; result: %s ;'%(index_no, data[int(index_no)])
#########Here##########
    
@app.route('/method/<string:methodname>')
def get_image_for_method(methodname):
    return f'Method {methodname}'

@app.route('/accuracy/<string:filename>')
def get_accuracy_for_image(filename):
    return f'Filename: {filename} <b>Drake</b> 21%, <b>Red-Breasted Merganser</b> 5%, <b>Maillot</b> 4%, <b>Prairie Chicken</b> 3%, <b>Leatherback Turtle</b> 3%" '

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join('images/test01/users/', filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory('images/test01/users/', name)

@app.route('/classify/<name>/<string:testnbr>')
def classify_image(name,testnbr):
    jsonData={}
    if os.path.exists("images/"+testnbr.lower()+"/static/"+name):
        img = Image.open("images/"+testnbr.lower()+"/static/"+name)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        #with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        jsonData={}
        for i in range(top5_prob.size(0)):
            prediction={}
            prob=top5_prob[i].item()
            prediction["probability"] = np.around(top5_prob[i].item(),2)
            prediction["label"] = categories[top5_catid[i]]
            prediction["labelid"] = top5_catid[i].item()
            jsonData[i]=prediction
    return jsonify(jsonData)

##FBA Method section

@app.route("/occlusion/<string:filename>/<int:label>/<string:testnbr>")
def occlusion(filename,label,testnbr):
    if (testnbr=="Test01"):
        flaskmethod="/fba-files/"
    if (testnbr=="Test02"):
        flaskmethod="/fba-files2/"
    if (testnbr=="Test03"):
        flaskmethod="/fba-files3/"
    print("folder:"+flaskmethod,file=sys.stderr)
    method_prefix="occlusion_"
    if not os.path.exists("images/"+testnbr.lower()+"/fba/"+method_prefix+filename):
        img=Image.open("images/"+testnbr.lower()+"/static/"+filename)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        occlusion = Occlusion(model)
        attributions = occlusion.attribute(input_batch,
                                  strides = (3, 25, 25),
                                  sliding_window_shapes=(3,50,50),
                                  target=label,
                                  baselines = 0)
        result = viz.visualize_image_attr(attributions[0].cpu().permute(1,2,0).detach().numpy(),
                                  transform(img).cpu().permute(1,2,0).numpy(),
                                  cmap=default_cmap,
                                  alpha_overlay=alpha_overlay,
                                  method=method,
                                  sign=sign) 
        result[0].savefig("images/"+testnbr.lower()+"/fba/"+method_prefix+filename,
                          bbox_inches="tight",
                          pad_inches=0);
    #Here folder is method to call to get the image
    return flaskmethod+method_prefix+filename

#label equals imagenet label
@app.route("/layer_gradcam/<string:filename>/<int:label>/<string:testnbr>")
def layer_gradcam(filename,label,testnbr):
    if (testnbr=="Test01"):
        flaskmethod="/fba-files/"
    if (testnbr=="Test02"):
        flaskmethod="/fba-files2/"
    if (testnbr=="Test03"):
        flaskmethod="/fba-files3/"
    print("folder:"+flaskmethod,file=sys.stderr)
    method_prefix="layer_gradcam_"
    if not os.path.exists("images/"+testnbr.lower()+"/fba/"+method_prefix+filename):
        img=Image.open("images/"+testnbr.lower()+"/static/"+filename)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        layer_gradcam = LayerGradCam(model, model.layer4[2].conv3)
        attributions = layer_gradcam.attribute(input_batch, label,relu_attributions=True)
        attributions = LayerAttribution.interpolate(attributions, input_batch.shape[2:])
        result = viz.visualize_image_attr(attributions[0].cpu().permute(1,2,0).detach().numpy(),
                             transform(img).cpu().permute(1,2,0).detach().numpy(), 
                             cmap=default_cmap,
                             alpha_overlay=alpha_overlay,
                             method=method,
                             sign=sign)
        result[0].savefig("images/"+testnbr.lower()+"/fba/"+method_prefix+filename,
                          bbox_inches="tight",
                          pad_inches=0);
    return flaskmethod+method_prefix+filename

@app.route("/guided_gradcam/<string:filename>/<int:label>/<string:testnbr>")
def guided_gradcam(filename,label,testnbr):
    if (testnbr=="Test01"):
        flaskmethod="/fba-files/"
    if (testnbr=="Test02"):
        flaskmethod="/fba-files2/"
    if (testnbr=="Test03"):
        flaskmethod="/fba-files3/"
    print("folder:"+flaskmethod,file=sys.stderr)
    method_prefix="guided_gradcam_"
    if not os.path.exists("images/"+testnbr.lower()+"/fba/"+method_prefix+filename):
        img=Image.open("images/"+testnbr.lower()+"/static/"+filename)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        guided_gradcam = GuidedGradCam(model, model.layer4[2].conv3)
        attributions = guided_gradcam.attribute(input_batch, label)
        result = viz.visualize_image_attr(attributions[0].cpu().permute(1,2,0).detach().numpy(),
                             transform(img).cpu().permute(1,2,0).detach().numpy(),
                             cmap=default_cmap,
                             alpha_overlay=alpha_overlay,
                             method=method,
                             sign=sign)
        result[0].savefig("images/"+testnbr.lower()+"/fba/"+method_prefix+filename,
                          bbox_inches="tight",
                          pad_inches=0);
    return flaskmethod+method_prefix+filename

@app.route("/integrated_gradients/<string:filename>/<int:label>/<string:testnbr>")
def integrated_gradients(filename,label,testnbr):
    if (testnbr=="Test01"):
        flaskmethod="/fba-files/"
    if (testnbr=="Test02"):
        flaskmethod="/fba-files2/"
    if (testnbr=="Test03"):
        flaskmethod="/fba-files3/"
    method_prefix="integrated_gradients_"
    if not os.path.exists("images/"+testnbr.lower()+"/fba/"+method_prefix+filename):
        img=Image.open("images/"+testnbr.lower()+"/static/"+filename)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        integrated_gradients = IntegratedGradients(model)
        attributions = integrated_gradients.attribute(input_batch, target=label)
        result = viz.visualize_image_attr(attributions[0].cpu().permute(1,2,0).detach().numpy(),
                             transform(img).permute(1,2,0).numpy(),
                             method=method,
                             cmap=default_cmap,
                             alpha_overlay=alpha_overlay,
                             sign=sign)
        result[0].savefig("images/"+testnbr.lower()+"/fba/"+method_prefix+filename,
                          bbox_inches="tight",
                          pad_inches=0);
    return flaskmethod+method_prefix+filename

@app.route("/gradient_shap/<string:filename>/<int:label>/<string:testnbr>")
def gradient_shap(filename,label,testnbr):
    if (testnbr=="Test01"):
        flaskmethod="/fba-files/"
    if (testnbr=="Test02"):
        flaskmethod="/fba-files2/"
    if (testnbr=="Test03"):
        flaskmethod="/fba-files3/"
    print("folder:"+flaskmethod,file=sys.stderr)
    method_prefix="gradient_shap_"
    if not os.path.exists("images/"+testnbr.lower()+"/fba/"+method_prefix+filename):
        img=Image.open("images/"+testnbr.lower()+"/static/"+filename)
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        gradient_shap = GradientShap(model)
        baselines = torch.randn(1, 3, 224, 224, requires_grad=True)
        if torch.cuda.is_available():
            baselines = baselines.to('cuda')
        attributions = gradient_shap.attribute(input_batch, baselines, target=label, n_samples=10)
        attributions = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0))
        result = viz.visualize_image_attr(attributions,preprocess(img).permute(1,2,0).numpy(),
                                  cmap=default_cmap,
                                  alpha_overlay=alpha_overlay,
                                  method=method,
                                  sign=sign)
        result[0].savefig("images/"+testnbr.lower()+"/fba/"+method_prefix+filename,
                          bbox_inches="tight",
                          pad_inches=0);
    return flaskmethod+method_prefix+filename

##Used for static files in user study1. (less than 255 bytes in name)
@app.route('/static-files/<name>')
def static_storage(name):
    return send_from_directory("images/test01/static/",name)

##Used for files created by methods in study1(less than 255 bytes in name)
@app.route('/fba-files/<name>')
def fba_storage(name):
    return send_from_directory("images/test01/fba/",name)

##Used for static files in user study2. (less than 255 bytes in name)
@app.route('/static-files2/<name>')
def static_storage2(name):
    return send_from_directory("images/test02/static/",name)

##Used for files created by methods in study2 (less than 255 bytes in name)
@app.route('/fba-files2/<name>')
def fba_storage2(name):
    return send_from_directory("images/test02/fba/",name)

##Used for static files in user study2. (less than 255 bytes in name)
@app.route('/static-files3/<name>')
def static_storage3(name):
    return send_from_directory("images/test03/static/",name)

##Used for files created by methods in study2 (less than 255 bytes in name)
@app.route('/fba-files3/<name>')
def fba_storage3(name):
    return send_from_directory("images/test03/fba/",name)

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
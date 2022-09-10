[Intotext for animal study](images/animals/0.PNG)

[![](testset/animalsthumb/4.jpg)](images/animals/1.PNG)
[![](testset/animalsthumb/3.jpg)](images/animals/2.PNG)
[![](testset/animalsthumb/7.jpg)](images/animals/3.PNG)
[![](testset/animalsthumb/10.jpg)](images/animals/4.PNG)
[![](testset/animalsthumb/1.jpg)](images/animals/5.PNG)
[![](testset/animalsthumb/9.jpg)](images/animals/6.PNG)

[Final questions](images/animals/7.PNG)

[Intotext for headgear study](images/headgear/0.PNG)

[![](testset/headthumb/1.jpg)](images/headgear/1.PNG)
[![](testset/headthumb/2.jpg)](images/headgear/2.PNG)
[![](testset/headthumb/3.jpg)](images/headgear/3.PNG)
[![](testset/headthumb/4.jpg)](images/headgear/4.PNG)
[![](testset/headthumb/5.jpg)](images/headgear/5.PNG)
[![](testset/headthumb/6.jpg)](images/headgear/6.PNG)
[![](testset/headthumb/7.jpg)](images/headgear/7.PNG)
[![](testset/headthumb/8.jpg)](images/headgear/8.PNG)
[![](testset/headthumb/9.jpg)](images/headgear/9.PNG)
[![](testset/headthumb/10.jpg)](images/headgear/10.PNG)

[Final questions](images/headgear/12.PNG)

```python
#Loading pretrained resnet50 model with V1 weights
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True #Skum grej från https://github.com/pytorch/pytorch/
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

#transformations
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

#For the vizualisation in captum
method="blended_heat_map"
sign="positive"
alpha_overlay = 0.6
default_cmap = LinearSegmentedColormap.from_list('custom green',
                                                [(0, '#39422c'),
                                                 (1, '#8df505')], N=5)

##Inference part
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
     #with torch.no_grad():
output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)


```                                           

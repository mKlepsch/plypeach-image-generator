# flask
from flask import Flask,render_template, request
import glob
import re

list_of_dcgan_images = {int(re.search(r'\d+', file).group(0)): file for file in glob.glob('./static/images/dcgan/*.png')}
list_of_dcgan_images = dict(sorted(list_of_dcgan_images.items()))
list_of_dcgan_images = {key:value for key, value in list_of_dcgan_images.items() if key in range(0,606)} 

list_of_ddpm_images = {int(re.search(r'\d+', file).group(0)): file for file in glob.glob('./static/images/ddpm/*.jpg')}
list_of_ddpm_images = dict(sorted(list_of_ddpm_images.items()))

app = Flask(__name__)

@app.route('/')
def landing_page():
    '''
    renders landing page
    '''
    return render_template('index.html')

@app.route('/generate')
def gallery():
    method = request.args.get("Method")
    if method =='DCGAN':
        return render_template('generate_2.html',gen_method=method,images=list_of_dcgan_images)
    elif method =='DDPM':
        return render_template('generate_2.html',gen_method=method,images=list_of_ddpm_images)
    else:
        render_template('generate_2.html',gen_method="You shouldnt be here",images=[])

@app.route('/about')
def about():
    return render_template('about.html')
        
if __name__=='__main__':
    # import torch
    # from ddmp import UNet, Diffusion
    # import matplotlib.pylab as plt

    #model_dcgan = "/mnt/data/spiced/final_project/dcgan/checkpoints/"
    #model_ddpm = "/mnt/data/spiced/final_project/ddmp_ply_peach/model/"
    #device ="cpu"
    ### load models
        ### load dcgan
    #latest_model_dcgan = newest_file(model_dcgan)
    # def load_dggan_model(path):
    #     pass
    
    #     ### load ddpm
    # #latest_model_ddpm = newest_file(model_ddpm)
    # def load_ddpm_model(path):
    #     ckpt = torch.load(path,map_location=torch.device(device))
    #     model = UNet(device=device).to(device)
    #     state_dict =ckpt['model']
    #     state_dict = {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys()}
    #     model.load_state_dict(state_dict)
    #     model = model.to(device)
    #     return model.cpu()
    

    # model = load_ddpm_model(path= latest_model_ddpm)
    # diffusion = Diffusion(img_size=64, device=device)
    # x = diffusion.sample(model, 1)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #                         torch.cat([i for i in x.cpu()], dim=-1),
    #                         ], dim=-2).permute(1, 2, 0).cpu())
    # plt.axis('off')
    # plt.show()
    list_of_dcgan_images = {int(re.search(r'\d+', file).group(0)): file for file in glob.glob('./static/images/dcgan/*.png')}
    list_of_dcgan_images = dict(sorted(list_of_dcgan_images.items()))
    list_of_dcgan_images = {key:value for key, value in list_of_dcgan_images.items() if key in range(0,606)} 
    
    list_of_ddpm_images = {int(re.search(r'\d+', file).group(0)): file for file in glob.glob('./static/images/ddpm/*.jpg')}
    list_of_ddpm_images = dict(sorted(list_of_ddpm_images.items()))
    
    app.run(debug=False,port=5000,host="0.0.0.0")
# lib gui
from tkinter import*
from PIL import Image,ImageTk
from tkinter import filedialog

#import lib lowl ight
import torch
import torch.nn.functional as F
import os
from runpy import run_path
from skimage import img_as_ubyte
import cv2

# star setup low light 

task = "lowlight_enhancement"

parameters = {
    'inp_channels': 3,
    'out_channels': 3,
    'n_feat': 80,
    'chan_factor': 1.5,
    'n_RRG': 4,
    'n_MRB': 2,
    'height': 3,
    'width': 2,
    'bias': False,
    'scale': 1,
    'task': task
}
weights = os.path.join('Enhancement', 'pretrained_models', 'enhancement_lol.pth')
load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
model = load_arch['MIRNet_v2'](**parameters)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

img_multiple_of = 4
# end star setup low light 
#=======================================================================

#desain gui
win = Tk()
win.geometry("600x700")
win.resizable(False, False)
win.configure(bg ='#1b407a')
w = 400
h = 300

color = "#581845"
frame_1 = Frame(win,width = 600,height =320,bg = color).place(x=0,y=0)
frame_2 = Frame(win,width = 600,height =320,bg = color).place(x=0,y=350)

v = Label(frame_1, width=w, height=h)
v.place(x=10, y=10)
#end desain gui

#webcam
cap = cv2.VideoCapture(0)

def Save():
    file = filedialog.asksaveasfilename(filetypes=[("PNG", ".png")])
    image = Image.fromarray(restored)
    image.save(file+'.png')
    print(file)

def take_copy(im):
    global restored

    la = Label(frame_2, width=w, height=h)
    la.place(x=10, y=355)
    #copy img for webcam
    copy = im.copy()    
    #proses
    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            input_ = torch.from_numpy(copy).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
            # Pad the input if not_multiple_of 8
            height, width = input_.shape[2], input_.shape[3]
            H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                    (width + img_multiple_of) // img_multiple_of) * img_multiple_of
            padh = H - height if height % img_multiple_of != 0 else 0
            padw = W - width if width % img_multiple_of != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)
            restored = restored[:, :, :height, :width]
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            restored = img_as_ubyte(restored[0])
       
        #end proses


    #view capture lowlight
    rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image)
    la.configure(image=imgtk)
    la.image = imgtk
    #button save
    save = Button(win,text = "save",command=lambda : Save())
    save.place(x=450,y=500, width=60, height=50)
    
#view webcam
def select_img():
    global img
    _, img = cap.read()
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image) # menapilkan gambar
    v.configure(image=imgtk)
    v.image = imgtk
    v.after(10, select_img)

#buttom capture
select_img()
snap = Button(win, text="capture", command=lambda: take_copy(img))
snap.place(x=450, y=150, width=60, height=50)

win.mainloop()

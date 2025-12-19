from tkinter import *
import tkinter as tk
import cv2
import os
import math
from tkinter import filedialog
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imutils
from PIL import Image
from PIL import ImageTk
from sklearn.model_selection import KFold
# global variables
bg = None
import time
from PIL import ImageTk, Image
from resizeimage import resizeimage
from skimage.filters import median
from FooderImage import *
import colorgram
import pandas as pd
global rep
import csv
import copy
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
def callback(selection):
    global xname
    xname=selection

def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list

def simple_query(full_path, as_gray=False):
    # simplify the creation of a query
    return FooderImage(full_path,
                       as_gray=as_gray,
                       as_pre_processed=True,
                       auto_compute_glcm=True,
                       auto_compute_color_moment=True)
    
def get_all_images(folder, ext):
    all_files = []
    # Iterate through all files in folder
    for file in os.listdir(folder):
        # Get the file extension
        _, file_ext = os.path.splitext(file)

        # If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    # Get list of all files
    return all_files

def histogram_feature_extraction(img_name):
    image = cv2.imread(img_name)
    chans = cv2.split(image)
    features = []
    hist1 = cv2.calcHist([chans[0]], [0], None, [170], [0, 256])
    hist2 = cv2.calcHist([chans[1]], [0], None, [170], [0, 256])
    hist3 = cv2.calcHist([chans[2]], [0], None, [170], [0, 256])
    features=np.concatenate((hist1,hist2,hist3))
    return features

def get_index_positions_2(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    for i in range(len(list_of_elems)):
        if list_of_elems[i] == element:
            index_pos_list.append(i)
    return index_pos_list


def simple_query1(full_path, as_gray=False):
    # simplify the creation of a query
    return FooderImage(full_path,
                       as_gray=as_gray,
                       as_pre_processed=True,
                       auto_compute_glcm=True,
                       auto_compute_color_moment=True)

def Color11(name):

    colors = colorgram.extract(name, 6)

    # colorgram.extract returns Color objects, which let you access
    # RGB, HSL, and what proportion of the image was that color.
    first_color = colors[0]
    rgb = first_color.rgb # e.g. (255, 151, 210)
    hsl = first_color.hsl # e.g. (230, 255, 203)
    return [rgb[0],rgb[1],rgb[2],hsl[0],hsl[1],hsl[2]]
    

def glcm11(name):
    q = simple_query1(name)
    f1 = q.get_feature_vector()
   
    return f1

    
class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master

        # changing the title of our master widget      
        self.master.title("Image Retival")
        
        self.pack(fill=BOTH, expand=1)
        w = tk.Label(root, 
		 text=" IMAGE RETRIVAL SYSTEM ",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 20 bold italic")
        w.pack()
        w.place(x=350, y=0)
        # creating a button instance
        quitButton = Button(self,command=self.query, text="LOAD IMAGE",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=50)
        quitButton = Button(self,command=self.preprocess,text="PREPROCESSING",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=300)
        quitButton = Button(self,command=self.segment, text="SEGMENTATION",fg="blue",activebackground="dark red",width=20)
        quitButton.place(x=10, y=350)
        quitButton = Button(self,command=self.feature,text="FEATURE EXTRACTION",activebackground="dark red",fg="blue",width=20)
        quitButton.place(x=10, y=400)
        quitButton = Button(self,command=self.Data_Processing,text="Data Processing",activebackground="dark red",fg="blue",width=20)
        quitButton.place(x=10, y=450)
        quitButton = Button(self,command=self.cbir_Euclidean_dist,text="CBIR Euclidean distance Test",activebackground="dark red",fg="blue",width=25)
        quitButton.place(x=10, y=500)
        quitButton = Button(self,command =self.cbir_ANN,text="CBIR ANN Test",activebackground="dark red",fg="blue",width=20)
        quitButton.place(x=10, y=550)
        
        load = Image.open("logo.png")
        render = ImageTk.PhotoImage(load)
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=10, y=90)

        load = Image.open("logo.png")
        render = ImageTk.PhotoImage(load)

        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=250, y=50)

        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=500, y=50)

        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image4.image = render
        image4.place(x=750, y=50)
        
#       2nd row

        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image5.image = render
        image5.place(x=250, y=270)

        image6=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image6.image = render
        image6.place(x=500, y=270)

        #image7=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        #image7.image = render
        #image7.place(x=750, y=270)

#       3rd column
        variable = StringVar(self)
        variable.set("SELECT FEATURE TYPE") # default value
        wi = OptionMenu(self, variable, "Color Features", "Histogram Feature", "GLCM Features","ALL Features",command=callback)
        wi.pack()
        wi.place(x=980, y=50)
        contents ="  Waiting for Results..."
        global T
        T = Text(self, height=19, width=25)
        T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        
#       3rd row
        #image5.place(x=300, y=490)

#       Functions

    def query(self, event=None):
        contents ="Loading Image..."
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[
            ("JPEG", "*.jpg"),
            ("JPEG", "*.jpeg")
        ])
        # Image operation using thresholding 
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(512,384))
        Input_img=img.copy()
        print(rep[0])
        img= cv2.resize(img,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img)
        render = ImageTk.PhotoImage(self.from_array)
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image1.image = render
        image1.place(x=10, y=90)
        #cv2.destroyAllWindows()
        contents="Image Loadeded successfully !!"
        
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        self.Input_img=Input_img
    def close_window(): 
        Window.destroy()
        
    def preprocess(self, event=None):
        global T,rep
        contents="Pre-Processing ..."
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        img = cv2.imread(rep[0])
        img = cv2.resize(img,(512,384))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray= median(gray)
        img= cv2.resize(gray,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img)
        render = ImageTk.PhotoImage(self.from_array)
        image2=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image2.image = render
        image2.place(x=250, y=50)
                
        contents="Pre-Processing completed successfully \n 1) Color Conversion \n 2) Median filter  "
                   
        T = Text(self, height=20, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
    def segment(self, event=None):
        contents ="Segmentation Processing..."
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        img_org = cv2.imread(rep[0])
        img_org = cv2.resize(img_org,(512,384))
        gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        gray= median(gray)


        #k-means clustring
        Z = gray.reshape((-1,3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K=3
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((gray.shape))

        img= cv2.resize(res2,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img)
        render = ImageTk.PhotoImage(self.from_array)
        image3=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image3.image = render
        image3.place(x=500, y=50)
        
        ret, thresh = cv2.threshold(res2, 0, 255, 
                            cv2.THRESH_BINARY_INV +
                            cv2.THRESH_OTSU)
         
        img= cv2.resize(thresh,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img)
        render = ImageTk.PhotoImage(self.from_array)
        image4=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image4.image = render
        image4.place(x=750, y=50)
        
        # Noise removal using Morphological 
        # closing operation 
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
        #                           kernel, iterations = 2) 
        # Background area using Dialation 
        bg = cv2.dilate(closing, kernel, iterations = 2) 
        #cv2.imshow('Morphological image',thresh)
        # Finding foreground area 
        dist_transform = cv2.distanceTransform(bg, cv2.DIST_L2, 0) 
        ret, fg = cv2.threshold(dist_transform, 0.02
                                * dist_transform.max(), 255, 0) 

        img1= cv2.resize(fg,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img1)
        render = ImageTk.PhotoImage(self.from_array)
        image5=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image5.image = render
        image5.place(x=250, y=270)
        fg = fg.astype(np.uint8)
                #print(np.mean(xc),np.mean(results[r[i],:,:,:]))

                # find the contours from the thresholded image
        contours, hierarchy = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # draw all contours
        image = cv2.drawContours(img_org, contours, -1, (0, 255, 0), 2)
        img1= cv2.resize(image,(200,200), interpolation = cv2.INTER_AREA)
        self.from_array = Image.fromarray(img1)
        render = ImageTk.PhotoImage(self.from_array)
        image6=Label(self, image=render,borderwidth=15, highlightthickness=5, height=150, width=150, bg='white')
        image6.image = render
        image6.place(x=500, y=270)

       
        #cv2.destroyAllWindows()
        contents="Segmentation completed successfully !! \n 1) K-Means Clustring \n 2) Thresholding Otsu's method  \n 3) Morphological operation \n 4) Selecting ROI"
        
        T = Text(self, height=20, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        
    def feature(self, event=None):
        contents ="Feature Extracting..."
        global T,rep,xname
        list1=[]
        xname=[]
        hf1=[]
        ch1=[]
        glcm1=[]
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(rep[0])
        
        hf1=histogram_feature_extraction(rep[0])
        hf1=(np.array(hf1)).reshape(510,1)
        hf1=np.uint8(hf1)
        
        ch=Color11(rep[0])
        cm=glcm11(rep[0])
        glcmf=[]
        ch.append(cm['mean'])
        ch.append(cm['variance'])
        ch.append(cm['skewness'])
        ch1=np.uint8((np.array(ch)).reshape(9,1))
        
        
        glcmf.append(cm['contrast'])
        glcmf.append(cm['dissimilarity'])
        glcmf.append(cm[ 'homogeneity'])
        glcmf.append(cm['ASM'])
        glcmf.append(cm['energy'])
        glcmf.append(cm['correlation'])
        glcm1=((np.array(glcmf)).reshape(6,1))
        glcm1=np.uint8(glcm1*100)

       
        print("Histogram Feature \n", np.resize(hf1,(1,510)))
        print("Color Feature \n", np.resize(ch1,(1,9)))
        print("GLCM Features \n",np.resize(glcm1,(1,6)))
        
        #cv2.destroyAllWindows()
        contents="Feature Extraction completed successfully !!"
        T = Text(self, height=19, width=25)
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)

        feature1 = np.concatenate((ch1,glcm1,hf1))
        feature1=list(np.ravel(feature1[:]))
        self.feature1=feature1
        

    
    def Data_Processing(self, event=None):
        contents ="DATA Processing..."
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print(contents)
        
        foldername_list=[]
        image_pathname=[]
        featurematrix=[]
        label=[]
        loc1=1
        count1=0
        cw_directory = os.getcwd()
        #folder='E:/H/Project 2020 December 16/Visual Information Retrieval/New code/dataset'
        folder=cw_directory+'/dataset'
        rep=[]
        images = []
        for filename in os.listdir(folder):

            sub_dir=os.path.join(folder,filename)
            print(os.path.join(folder,filename))
            
            for img_name in os.listdir(sub_dir):
                hf1=[]
                ch1=[]
                glcm1=[]
                rep=os.path.join(sub_dir, img_name)
                image_pathname.append(rep)
                
                hf1=histogram_feature_extraction(rep)
                hf1=(np.array(hf1)).reshape(510,1)
                hf1=np.uint8(hf1)
                
                ch=Color11(rep)
                cm=glcm11(rep)
                glcmf=[]
                ch.append(cm['mean'])
                ch.append(cm['variance'])
                ch.append(cm['skewness'])
                ch1=np.uint8((np.array(ch)).reshape(9,1))
                
                
                glcmf.append(cm['contrast'])
                glcmf.append(cm['dissimilarity'])
                glcmf.append(cm[ 'homogeneity'])
                glcmf.append(cm['ASM'])
                glcmf.append(cm['energy'])
                glcmf.append(cm['correlation'])
                glcm1=((np.array(glcmf)).reshape(6,1))
                glcm1=np.uint8(glcm1*100)
                
                feature1 = np.concatenate((ch1,glcm1,hf1))
                feature1=list(np.ravel(feature1[:]))
                featurematrix.append(feature1)
                label.append(count1)
                loc1+=1
            count1+=1
            foldername_list.append(filename)   
        with open("featurematrix.csv", 'w') as f:
            for s in featurematrix:
                f.write(str(s) + '\n')
                
        with open("lablematrix.csv", 'w') as f:
            for s in label:
                f.write(str(s) + '\n')
                
        self.featurematrix=featurematrix
        self.label=label
        self.cw_directory=cw_directory
        self.image_pathname=image_pathname
        contents ="Data Processing Completed"
        self.foldername_list=foldername_list 
        T = Text(self, height=19, width=25)
        xx=[250,350,450,550,650,750,850,950,1050,1150]
        self.xx=xx
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        print("contents")
        
    def cbir_Euclidean_dist(self, event=None):
        image_pathname=self.image_pathname
        orgimg = self.from_array
        label= self.label
        cw_directory=self.cw_directory
        folder=cw_directory+'/dataset'
        foldername=os.listdir(folder)
        contents="TESTING QUERY...."
        xx=[250,350,450,550,650,750,850,950,1050,1150]
        global T,rep
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)
        #print(contents)
        img_feature=self.feature1
        featurematrix=self.featurematrix
        first_10minval=[]
        Euclidean_distance=[]
        for data_feature in featurematrix:
            Euclidean_dist = np.linalg.norm(np.array(img_feature) - np.array(data_feature))
            Euclidean_distance.append(Euclidean_dist)

        minimum_dist= Euclidean_distance.index(min(Euclidean_distance))
        fld_val= label[minimum_dist]
        first_10minval= Euclidean_distance.copy()
        first_10minval.sort(reverse=False)
        #print(label)
        itr1=0
        for i in first_10minval:
            val_id =Euclidean_distance.index(i)
            x1=xx[itr1]
            img_sort = cv2.imread(image_pathname[val_id])
            img= cv2.resize(img_sort,(100,100), interpolation = cv2.INTER_AREA)
            self.from_array = Image.fromarray(img)
            render = ImageTk.PhotoImage(self.from_array)
            image4=Label(self, image=render,borderwidth=5, highlightthickness=5, height=90, width=90, bg='white')
            image4.image = render
            image4.place(x=x1, y=500)
            itr1+=1
            if itr1==10:
                break
            #
        contents="QUERY Completed \n" + str(first_10minval)
        T = Text(self, height=19, width=25)
        #T.pack()
        T.place(x=950, y=150)
        T.insert(END,contents)

    def cbir_ANN(self, event=None):
        
        xx=[250,350,450,550,650,750,850,950,1050,1150]
        global T,rep
        T = Text(self, height=19, width=25)
        mage_pathname=self.image_pathname     
        feature1=self.feature1
        image_pathname=self.image_pathname
        orgimg = self.from_array
        featurematrix=self.featurematrix
        label= self.label
        cw_directory=self.cw_directory
        
        #int1=input("Want to Train Enter 1")
        int1=1
        if int1==1:
            model = MLPClassifier(activation='relu', verbose=True,
                                           hidden_layer_sizes=(100,), batch_size=64)
            model=model.fit(np.array(featurematrix),np.array(label))
        # save
            joblib.dump(model, "Trained_Model.pkl")
        else:
            filename="Trained_Model.pkl"
            model = joblib.load(filename)
        
        feature1=np.array(feature1)
        feature1=np.resize(feature1,(1,525))
        C_pred =model.predict_proba(np.array(feature1))
        C_pred=np.argmax(C_pred)
        
        #index_pos_list = get_index_positions_2(str(label), str(Y_pred[0]))
        #index_pos_list
        
        plot_confusion_matrix(model, featurematrix, label)
        plt.show()
        f_acc =model.predict(np.array(featurematrix))
        acc = accuracy_score(f_acc,label)
        plt.bar(['Neural networks'],[acc], label="ANN", color='g')
        plt.show()
        
        Y_pred =model.predict_proba(np.array(featurematrix))
        prediction1=[]
        for ypre1 in Y_pred:
            idx=ypre1[C_pred]
            prediction1.append(ypre1[C_pred])

        prediction2=prediction1.copy()
        print(prediction2)
        ###accending order shorting
        prediction1.sort(reverse=True)
        itr1=0
        idx=0
        index_pos_list = get_index_positions_2(prediction2,prediction1[0])
        
        for s_ind in index_pos_list:
            x1=xx[itr1]
            img_sort = cv2.imread(image_pathname[s_ind])
            img= cv2.resize(img_sort,(100,100), interpolation = cv2.INTER_AREA)
            self.from_array = Image.fromarray(img)
            render = ImageTk.PhotoImage(self.from_array)
            image4=Label(self, image=render,borderwidth=5, highlightthickness=5, height=90, width=90, bg='white')
            image4.image = render
            image4.place(x=x1, y=500)
            indx=idx
            #print(idx)
            itr1+=1
            if itr1==10:
                break


        if itr1<10:  
            for predmax in prediction1[itr1:]:
                x1=xx[itr1]
                idx=prediction2.index(predmax)
                
                if itr1==0:
                    indx=idx
                elif idx==indx:
                    idx+=1;
                    indx==idx
                else:
                    indx=idx
                img_sort = cv2.imread(image_pathname[idx])
                img= cv2.resize(img_sort,(100,100), interpolation = cv2.INTER_AREA)
                self.from_array = Image.fromarray(img)
                render = ImageTk.PhotoImage(self.from_array)
                image4=Label(self, image=render,borderwidth=5, highlightthickness=5, height=90, width=90, bg='white')
                image4.image = render
                image4.place(x=x1, y=500)
                print(idx,indx)
                itr1+=1
                if itr1==10:
                    break
     
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()

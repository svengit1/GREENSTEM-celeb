import os
import time

from PIL import Image as Img
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel

from loader import process
from plotting import CelebADrawer


class User:

    def __init__(self):
        self.windows = {1: "main", 2: "second"}
        self.last_screen = 1

    def set_last_screen(self, last_screen):
        self.last_screen = last_screen

    def get_last_screen(self):
        return self.last_screen

    def save_temp_image(self, image):
        self.image = image

    def load_temp_image(self):
        return self.image

    def __getitem__(self, item):
        return self.windows.get(item, 1)


class BasicWidgetFunctions:

    def goto_window(self, window_id):
        U.set_last_screen(self.manager.current)
        self.manager.current = U[window_id]
        self.superWindowInit(U[window_id])

    def goto_last(self):
        self.manager.current = U.get_last_screen()
        self.superWindowInit(self.manager.current)

    def getWindowSize(self):
        return Window.size

    def getScreen(self, screen_name):
        return self.manager.get_screen(screen_name)

    def superWindowInit(self, window_name):
        # All init functions must contain this function!
        new_screen = self.manager.get_screen(window_name)
        new_screen.start_init()


class MainWindow(Screen, BasicWidgetFunctions):
    def start_init(self):
        pass

    def capture_image(self):
        camera = self.ids["camera"]
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("temp_image.png")
        image = "temp_image.png"
        opener = Img.open(image)
        opener.resize((1920,1080)).save(image)
        print("Captured")
        U.save_temp_image(image)
        self.goto_window(2)


class SecondWindow(Screen, BasicWidgetFunctions):
    img_src = StringProperty("")
    processed_img_src = StringProperty("")
    landmarked_img_src = StringProperty("")
    bottom_text = StringProperty("Done!, you look like the person on the right..")

    def start_init(self):
        self.img_src = U.load_temp_image()
        self.box_logs = self.ids["boxTwo"]
        file, image_new, bbox, feats0,feats1,race,gender,age,inapprop = process("", self.img_src)
        D.process_img(base=False,image=Img.open(self.img_src),bbox=bbox,save=True)
        D.process_feats(img=image_new,features=feats0)
        self.processed_img_src = "processed_bbox.png"
        self.landmarked_img_src = "processed_feats.png"
        if inapprop:
            self.log_label = MDLabel(text=f"match is {file}, \n it is most probably inappropriate")
        else:
            self.log_label = Image(source="../img_celeba/"+file)
        self.box_logs.add_widget(self.log_label)
        self.bottom_text = f"You are {gender} and probably {race} and {age}"

    def go_back(self):
        os.remove(self.img_src)
        os.remove(self.landmarked_img_src)
        os.remove(self.processed_img_src)
        for key in range(1,4):
            self.img_one = self.ids[f"image{key}"]
            self.img_one.remove_from_cache()
        self.img_src = ""
        self.landmarked_img_src = ""
        self.processed_img_src= ""
        self.box_logs.remove_widget(self.log_label)
        self.goto_last()


class WindowManager(ScreenManager):
    pass


class MainApp(MDApp):
    def build(self):
        Start_Screen = Builder.load_file("core.kv")
        return Start_Screen


D = CelebADrawer.PillowManipulator()
U = User()
X = MainApp()
X.run()

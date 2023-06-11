import os

import pandas as pd
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.app import MDApp

from plotting import CelebADrawer

img_dir = os.getcwd().split("\\")[:-1]
new_dir = ""
for i in img_dir:
    new_dir += i + "\\\\"
img_dir = new_dir + "img_celeba\\\\"
print(img_dir)


class User:

    def __init__(self):
        self.windows = {1: "main", 2: "second"}
        self.last_screen = 1

    def set_last_screen(self, last_screen):
        self.last_screen = last_screen

    def get_last_screen(self):
        return self.last_screen

    def __getitem__(self, item):
        return self.windows.get(item, 1)


class ImagerUser(User):

    def read_assignment(self, name):
        self.name = name
        self.df = pd.read_csv("logs")
        print(self.df.columns)
        assignments = self.df.loc[self.df["Name"] == name]
        assignments = assignments.loc[assignments["Complete"] == 0]
        if not assignments.empty:
            assignments.sort_values(by="AID")
            aid, start,goal, current = assignments.iloc[0]["AID"], \
                                int(assignments.iloc[0]["Assigned"].split("-")[0]), \
                                int(assignments.iloc[0]["Assigned"].split("-")[1]), assignments.iloc[0]["Current"]
            self.aid,self.start, self.goal, self.current = aid,start, goal, current
            return aid, goal, current
        raise BaseException("All assigments complete, report to your manager!")
    def mod_assignment(self):
        if not self.id < len(self.df_classes):
            x = self.df.iloc[self.aid]
            if x["Current"]<self.id:
                x["Current"] = self.id
            print(x)
            self.df.iloc[self.aid] = x
            self.df.to_csv("logs", index=False)
        if self.id >= int(self.goal):
            x = self.df.iloc[self.aid]
            x["Complete"] = 1
            self.df.iloc[self.aid] = x
            self.df.to_csv("logs", index=False)

    def init_imgs(self):
        if not f"SkinLabels_{self.aid}_{self.name.replace(' ','_')}" in os.listdir():
            file = open(f"SkinLabels_{self.aid}_{self.name.replace(' ','_')}", "w", encoding="utf-8")
            file.write("ID,Name,Dark_Skin")
            file.close()
        self.df_classes = pd.read_csv(f"SkinLabels_{self.aid}_{self.name.replace(' ','_')}")
        last_id = self.df_classes.iloc[len(self.df_classes) - 1]["ID"] + 1 if not self.df_classes.empty else self.start
        self.id = last_id

    def add_img(self, label, name):
        if self.id-self.start<len(self.df_classes):
            x = self.df_classes.iloc[self.id-self.start]
            x["Dark_Skin"] = label
            self.df_classes.iloc[self.id-self.start] = x
        else:
            self.df_classes = self.df_classes._append(pd.DataFrame([{"ID": self.id, "Name": name, "Dark_Skin": label}]))
        self.id += 1
        self.df_classes.to_csv(f"SkinLabels_{self.aid}_{self.name.replace(' ','_')}", index=False)
    def get_cur_class(self):
        if self.id-self.start<len(self.df_classes):
            x = self.df_classes.iloc[self.id-self.start]
            dict = {1:"Dark",0:"Not Dark"}
            return dict[x["Dark_Skin"]]
        else:
            return None

    def save_temp_image(self, image):
        self.image = image

    def load_temp_image(self):
        return self.image


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
    helper_text = StringProperty("dobar dan")

    def __init__(self, **kw):
        super().__init__(**kw)

    def validate(self, widget):
        print("name added to registry")
        if widget.text not in ("Gabriel Janđel", "Sven Matković","Saida Deljac"):
            self.helper_text = "Error, invalid name/surname, format is Sven Matković"
            widget.text = ":("
        else:
            self.aid, self.goal, self.current = U.read_assignment(widget.text)
            self.goto_window(2)


class SecondWindow(Screen, BasicWidgetFunctions):
    image_addr = StringProperty("")
    progress = StringProperty("")
    infos = StringProperty("")

    def __init__(self, **kw):
        super().__init__(**kw)

    def close(self):
        self._keyboard.unbind(on_key_down=self.press)
        self._keyboard = None

    def press(self, keyboard, keycode, text, modifiers):
        print(keycode[1], keyboard, text, modifiers)
        print(self.events)
        try:self.events[keycode[1]]()
        except:print("error, bad key")

    def post(self, code):
        print(f"posting: {code}")
        U.add_img(code,self.Iname)
        U.mod_assignment()
        self.place_image(U.id)

    def undo(self):
        if not U.id-U.start: return
        U.id -= 1
        self.place_image(U.id)
    def fwd(self):
        if U.id-U.start>=len(U.df_classes): return
        U.id += 1
        self.place_image(U.id)


    def place_image(self,id):
        self.img_one = self.ids[f"image_holder"]
        self.img_one.remove_from_cache()
        image_name = "0"*(6-len(str(id+1))) + str(id+1) + ".jpg"
        self.image_addr = img_dir + image_name
        self.Iname= image_name
        self.progress = "Progress: " +str(round(((U.id-U.start)/(int(U.goal)-U.start))*100)) + "%"
        self.infos = f"Current image: {self.Iname} \n Current class: {U.get_cur_class()} \n Total images: {U.goal}"
        print(int(U.goal))
        print(int(U.id))
        if U.id/int(U.goal) >= 1:
            self.progress = "done!, good job, disabling controls..."
            self.events = {}
            self.close()

    def start_init(self):
        self.events = {"y": lambda: self.post(1), "n": lambda: self.post(0), "b": self.undo, "f":self.fwd}
        self._keyboard = Window.request_keyboard(self.close,
                                                 self)
        U.init_imgs()
        self._keyboard.bind(on_key_down=self.press)
        self.place_image(U.id)



class WindowManager(ScreenManager):
    pass


class MainApp(MDApp):
    def build(self):
        Start_Screen = Builder.load_file("classifier_core.kv")
        return Start_Screen


D = CelebADrawer.PillowManipulator()
U = ImagerUser()
X = MainApp()
X.run()

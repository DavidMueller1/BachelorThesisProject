from tkinter import *
from combine_results_util import compare_results, combine_results
from logger import Logger


class AnalyseApp:

    def __init__(self, master):
        self.master = master
        master.title("Analyser")

        self.label = Label(master, text="What would you like to do?")
        self.label.pack()

        self.separator_1 = Frame(master, height=1, width=250, bg="black")
        self.separator_1.pack(pady=10)

        self.analyse_button = Button(master, text="Compare results", command=self.compare)
        self.analyse_button.pack()

        self.custom_titles = IntVar()
        self.custom_titles_checkbox = Checkbutton(master, text="Custom titles", variable=self.custom_titles)
        self.custom_titles_checkbox.pack()

        self.separator_2 = Frame(master, height=1, width=250, bg="black")
        self.separator_2.pack(pady=10)

        self.combine_button = Button(master, text="Get average from multiple results", command=self.combine)
        self.combine_button.pack()

        self.combine_title = Entry(master)
        self.combine_title.pack()

    def compare(self):
        compare_results(custom_titles=(self.custom_titles.get() == 1))

    def combine(self):
        combine_results(title=self.combine_title.get())


root = Tk()
root.geometry("300x200")
my_gui = AnalyseApp(root)
root.mainloop()

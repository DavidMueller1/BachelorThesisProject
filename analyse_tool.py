from tkinter import *
from combine_results_util import compare_results, combine_results, compare_massive_results
from show_experiment_result import show_experiment_result
from logger import Logger


class AnalyseApp:

    def __init__(self, master):
        self.master = master
        master.title("Analyser")

        self.label = Label(master, text="What would you like to do?")
        self.label.pack()


        self.separator_1 = Frame(master, height=1, width=250, bg="black")
        self.separator_1.pack(pady=10)

        self.show_result_button = Button(master, text="Show result", command=self.show_result)
        self.show_result_button.pack()

        self.show_result_title = Entry(master)
        self.show_result_title.pack()

        self.random_spawn = IntVar()
        self.random_spawn_checkbox = Checkbutton(master, text="Random spawn", variable=self.random_spawn)
        self.random_spawn_checkbox.pack()


        self.separator_2 = Frame(master, height=1, width=250, bg="black")
        self.separator_2.pack(pady=10)

        self.analyse_button = Button(master, text="Compare results", command=self.compare)
        self.analyse_button.pack()

        self.custom_titles = IntVar()
        self.custom_titles_checkbox = Checkbutton(master, text="Custom titles", variable=self.custom_titles)
        self.custom_titles_checkbox.pack()


        self.separator_4 = Frame(master, height=1, width=250, bg="black")
        self.separator_4.pack(pady=10)

        self.analyse_massive_button = Button(master, text="Compare massive results", command=self.compare_massive)
        self.analyse_massive_button.pack()

        self.custom_titles_massive = IntVar()
        self.custom_titles_massive_checkbox = Checkbutton(master, text="Custom titles", variable=self.custom_titles_massive)
        self.custom_titles_massive_checkbox.pack()


        self.separator_3 = Frame(master, height=1, width=250, bg="black")
        self.separator_3.pack(pady=10)

        self.combine_button = Button(master, text="Get average from multiple results", command=self.combine)
        self.combine_button.pack()

        self.combine_title = Entry(master)
        self.combine_title.pack()

    def show_result(self):
        show_experiment_result(random_spawn=(self.random_spawn.get() == 1))

    def compare(self):
        compare_results(custom_titles=(self.custom_titles.get() == 1))

    def compare_massive(self):
        compare_massive_results(custom_titles=(self.custom_titles_massive.get() == 1))

    def combine(self):
        combine_results(title=self.combine_title.get())


root = Tk()
root.geometry("300x400")
my_gui = AnalyseApp(root)
root.mainloop()

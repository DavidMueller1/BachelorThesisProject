import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
screen = app.screens()[1]
dpi = screen.physicalDotsPerInch()
print(dpi)
app.quit()

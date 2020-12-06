
import sys
from PyQt5.QtWidgets import QApplication

from GUI import Window

if  __name__=='__main__':

    app = QApplication(sys.argv)

    window = Window()
    window.show()#transfer the window in the GUI



    sys.exit(app.exec_())

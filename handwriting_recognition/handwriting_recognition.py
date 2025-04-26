import os
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class HandwritingRecognition:
    def __init__(self, custom_data=False):
        self.black_pixel = np.full((1, 3), 0, dtype=np.uint8)
        self.white_pixel = np.full((1, 3), 255, dtype=np.uint8)
        self.test_sample = np.full((140, 140, 3), self.black_pixel, dtype=np.uint8)
        self.X, self.y = self._fetch_custom_data() if custom_data else self._fetch_mnist() 
        self.rf, self.knn, self.lr = self._train_models()
        self.last_x, self.last_y = None, None

    def _fetch_mnist(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetches the MNIST dataset from OpenML."""
        mnist = None
        if os.path.exists("mnist_784.csv"):
            mnist = np.loadtxt("mnist_784.csv", delimiter=",")
            X = mnist[:, 1:].astype(np.uint8)[:5000] # limit to speed up training
            y = mnist[:, 0].astype(np.uint8)[:5000]
        else:
            mnist = fetch_openml("mnist_784", parser="auto", version=1, as_frame=False)
            X = mnist.data.astype(np.uint8)
            y = mnist.target.astype(np.uint8)
            np.savetxt("mnist_784.csv", np.column_stack((y, X)), delimiter=",", fmt='%d')
        return X, y

    def _fetch_custom_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetches custom data from a CSV file."""
        X, y = None, None
        if os.path.exists("custom_data.csv"):
            data = np.loadtxt("custom_data.csv", delimiter=",")
            X = data[:, 1:].astype(np.uint8)
            y = data[:, 0].astype(np.uint8)
        return X, y

    def _train_models(self) -> tuple[RandomForestClassifier, KNeighborsClassifier, LogisticRegression]:
        """Finds best hyperparameters and trains models on the MNIST dataset.
        Saves the models to disk if they don't exist.
        Loads the models from disk if they do exist.
        """
        if "rf_model.pkl" in os.listdir():
            with open("rf_model.pkl", "rb") as f:
                rf = pickle.load(f)
        else:
            rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
            rf.fit(self.X, self.y)
            with open("rf_model.pkl", "wb") as f:
                pickle.dump(rf, f)

        if "knn_model.pkl" in os.listdir():
            with open("knn_model.pkl", "rb") as f:
                knn = pickle.load(f)
        else:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(self.X, self.y)
            with open("knn_model.pkl", "wb") as f:
                pickle.dump(knn, f)

        if "lr_model.pkl" in os.listdir():
            with open("lr_model.pkl", "rb") as f:
                lr = pickle.load(f)
        else:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(self.X, self.y)
            with open("lr_model.pkl", "wb") as f:
                pickle.dump(lr, f)

        return rf, knn, lr

    def canvas_to_array(self, drawspace: tk.Canvas) -> tk.Image:
        """Captures, grayscales, and resizes the image drawn on the canvas."""
        x = drawspace.winfo_rootx()
        y = drawspace.winfo_rooty()
        width = drawspace.winfo_width()
        height = drawspace.winfo_height()

        # grab and grayscale image
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        image = image.convert("L")
        image = ImageOps.invert(image) # invert tkinter black and white
        image = image.resize((28, 28), Image.Resampling.LANCZOS) # high quality resizing

        # display the captured image
        new_window = tk.Toplevel()
        new_window.title("Captured Image")
        new_window.geometry("200x200")
        img = ImageTk.PhotoImage(image.resize((200, 200), Image.Resampling.LANCZOS))
        panel = tk.Label(new_window, image=img)
        panel.image = img  # keep a reference to avoid garbage collection
        panel.pack()

        numpy_array = np.array(image)
        numpy_array = (numpy_array > 128).astype(np.uint8) * 255  # 0 black, 255 white
        numpy_array = numpy_array.reshape(-1, 784)
        return numpy_array

    def guess_digit(self, drawspace: tk.Canvas, main_pred_lbl: tk.Label, pred_lbls: list[tk.Label]=None) -> None:
        """Guesses the digit drawn on the canvas and displays the prediction."""
        numpy_array = self.canvas_to_array(drawspace)

        pred_rf = self.rf.predict(numpy_array)[0]
        pred_knn = self.knn.predict(numpy_array)[0]
        pred_lr = self.lr.predict(numpy_array)[0]

        print(f"Random Forest: {pred_rf}")
        print(f"K Nearest Neighbors: {pred_knn}")
        print(f"Logistic Regression: {pred_lr}")
        
        # boyer-moore voting algorithm to determine majority vote
        count = 0
        candidate = None
        for num in [pred_rf, pred_knn, pred_lr]:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1

        main_pred_lbl.config(text=f"Prediction: {candidate}")
        if pred_lbls:
            pred_lbls[0].config(text=f"Random Forest: {pred_rf}")
            pred_lbls[1].config(text=f"K Nearest Neighbors: {pred_knn}")
            pred_lbls[2].config(text=f"Logistic Regression: {pred_lr}")

    def add_custom_data(self, drawspace: tk.Canvas, custom_data_combobox: ttk.Combobox) -> None:
        """Adds custom data to the CSV file."""
        numpy_array = self.canvas_to_array(drawspace)

        with open("custom_data.csv", "a") as f:
            f.write(f"{custom_data_combobox.get()},{','.join(map(str, numpy_array[0]))}\n")

    def clear_drawing(self, drawspace: tk.Canvas) -> None:
        """Clears the drawing canvas and internal array."""
        drawspace.delete("all")  # clear canvas
        self.test_sample[:] = self.black_pixel  # internal representation of drawing

    def start_drawing(self, event: tk.Event) -> None:
        """Sets the initial coordinates of the mouse when drawing."""
        self.last_x = event.x
        self.last_y = event.y

    def enable_multiple_pred(self, drawspace: tk.Canvas, main_pred_lbl: tk.Label, pred_lbls: list[tk.Label], multiple_pred_checkbox: ttk.Checkbutton, guess_btn: tk.Button) -> None:
        """Enables or disables the display of multiple predictions."""
        if multiple_pred_checkbox.instate(["selected"]):
            guess_btn.config(command=lambda: self.guess_digit(drawspace, main_pred_lbl, pred_lbls))
        else:
            for pred_lbl in pred_lbls:
                pred_lbl.config(text="")
            guess_btn.config(command=lambda: self.guess_digit(drawspace, main_pred_lbl))

    def draw(self, event: tk.Event, drawspace: tk.Canvas) -> None:
        """Draws on the canvas and internal array based on mouse coordinates."""
        # xy-coordinates and radius of the circle being drawn
        drawspace.create_line(
            (self.last_x, self.last_y, event.x, event.y),
            width=10,
            fill="black",
            capstyle=tk.ROUND,
            smooth=tk.TRUE,
        )
        self.test_sample[event.y - 2 : event.y + 3, event.x - 2 : event.x + 3] = (
            self.white_pixel
        )
        self.last_x = event.x
        self.last_y = event.y

    def draw_window(self) -> None:
        """Creates the drawing window and all its components."""
        window = tk.Tk()
        window.geometry("400x460")
        window.wm_title("Drawing Canvas")

        drawspace = tk.Canvas(
            window,
            width=200,
            height=200,
            bg="white",
            cursor="tcross",
            highlightthickness=1,
            highlightbackground="steelblue",
        )

        title_lbl = tk.Label(window, text="Draw a digit", font=("Helvetica", 16))
        main_pred_lbl = tk.Label(window, text="Prediction: ", font=("Helvetica", 14))
        rf_pred_lbl = tk.Label(window, text="", font=("Helvetica", 10), state="disabled")
        knn_pred_lbl = tk.Label(window, text="", font=("Helvetica", 10), state="disabled")
        lr_pred_lbl = tk.Label(window, text="", font=("Helvetica", 10), state="disabled")

        multiple_pred_checkbox = ttk.Checkbutton(window, text="Show Multiple Predictions", variable=tk.BooleanVar(), command=lambda: self.enable_multiple_pred(drawspace, main_pred_lbl, [rf_pred_lbl, knn_pred_lbl, lr_pred_lbl], multiple_pred_checkbox, guess_btn))
        
        custom_data_combobox = ttk.Combobox(window, values=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], state="readonly")
        custom_data_combobox.set("0")
        
        clear_btn = tk.Button(
            window, text="Clear", command=lambda: self.clear_drawing(drawspace)
        )
        guess_btn = tk.Button(
            window, text="Guess", command=lambda: self.guess_digit(drawspace, main_pred_lbl, [rf_pred_lbl, knn_pred_lbl, lr_pred_lbl])
        )
        add_custom_data_btn = tk.Button(
            window, text="Add Custom Data", command=lambda: self.add_custom_data(drawspace, custom_data_combobox)
        )

        drawspace.bind("<Button-1>", self.start_drawing)
        drawspace.bind("<B1-Motion>", lambda event: self.draw(event, drawspace))

        title_lbl.pack()
        drawspace.pack()
        main_pred_lbl.pack()
        guess_btn.pack(pady=2)
        clear_btn.pack(pady=2)
        custom_data_combobox.pack()
        add_custom_data_btn.pack()
        multiple_pred_checkbox.pack()
        rf_pred_lbl.pack()
        knn_pred_lbl.pack()
        lr_pred_lbl.pack()

        window.resizable(False, False)
        window.mainloop()


if __name__ == "__main__":
    hr = HandwritingRecognition(custom_data=False)
    hr.draw_window()

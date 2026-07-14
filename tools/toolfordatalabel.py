
import os
import json
import time
import tkinter as tk
from tkinter import messagebox, ttk, font as tkfont, filedialog
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
# --- Import latex generator ---
import gen_latex


CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1200
PREVIEW_MIN_SIZE = 240
PREVIEW_MAX_SIZE = 600
DEFAULT_PEN_WIDTH = 3
DEFAULT_PREVIEW_SIZE = 400
OUTPUT_SUBDIR = "inks"
PROGRESS_FILE = "progress.json"
GEN_IMG_DIR = "Data/GenData"
GEN_IMG_NUM = 20  # Số lượng ảnh latex muốn sinh ra mỗi lần chạy
GEN_RECORDS = []

INSTRUCTION_TEXT = (
    "Shortcuts: Ctrl+S save & next - Ctrl+Z undo - Ctrl+Shift+Z clear - "
    "Ctrl+Right skip - Ctrl+Left previous - F1 help."
)


image_list = []
latex_list = []
current_index = 0
current_image_path = ""
current_latex = ""
output_dir = OUTPUT_SUBDIR
strokes = []
pen_color = "black"
pen_width = DEFAULT_PEN_WIDTH
preview_size = max(PREVIEW_MIN_SIZE, min(PREVIEW_MAX_SIZE, DEFAULT_PREVIEW_SIZE))


root = None
img_tk = None
status_var = None
progress_var = None
image_name_var = None
folder_var = None
pen_value_var = None
preview_size_var = None
preview_size_value_var = None
progress_bar = None
canvas = None
image_panel = None
pen_slider = None
preview_slider = None
color_buttons = []

def load_progress(folder_path):
    if not folder_path:
        return 0
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}
    return data.get(folder_path, 0)

def save_progress(folder_path, index):
    if not folder_path:
        return
    data = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[folder_path] = index
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass

def update_status(message):
    if status_var is not None:
        status_var.set(message)

def confirm_discard():
    if not strokes:
        return True
    return messagebox.askyesno(
        "Discard strokes?",
        "You have unsaved strokes on this image. Leaving now will lose them.\nContinue?",
        parent=root,
    )

def update_canvas_scrollregion():
    if canvas is None:
        return
    bbox = canvas.bbox("all")
    if bbox is None:
        canvas.configure(scrollregion=(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT))
    else:
        x0, y0, x1, y1 = bbox
        x0 = min(0, x0)
        y0 = min(0, y0)
        x1 = max(x1, CANVAS_WIDTH)
        y1 = max(y1, CANVAS_HEIGHT)
        canvas.configure(scrollregion=(x0, y0, x1, y1))

def reset_canvas_view():
    if canvas is None:
        return
    canvas.xview_moveto(0)
    canvas.yview_moveto(0)




# --- Tự động sinh ảnh latex, chọn folder lưu và nạp vào app ---
def auto_generate_images():
    global image_list, latex_list, current_index, output_dir, save_folder, gen_records
    if not confirm_discard():
        return
    # Chọn folder lưu
    folder = filedialog.askdirectory(title="Chọn thư mục để lưu dữ liệu")
    if not folder:
        update_status("Bạn chưa chọn thư mục lưu.")
        return
    save_folder = folder
    img_dir = os.path.join(save_folder, "gen_images")
    os.makedirs(img_dir, exist_ok=True)
    output_dir = os.path.join(save_folder, OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    # Sinh ảnh latex mới
    records = gen_latex.generate_latex_images(GEN_IMG_NUM, img_dir)
    image_list = [rec["file_name"] for rec in records]
    latex_list = [rec["latex"] for rec in records]
    gen_records = records
    # Lưu records vào file json
    json_path = os.path.join(save_folder, "latex_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    folder_var.set(f"Đã sinh {len(image_list)} latex images vào: {save_folder}")
    current_index = 0
    progress_bar["maximum"] = max(len(image_list), 1)
    goto_index(current_index)
    update_status(f"Generated {len(image_list)} latex images. Start drawing when ready.")


def show_image():
    global img_tk, strokes, current_image_path, current_latex
    if not image_list:
        return
    total = len(image_list)
    if current_index >= total:
        show_completion()
        return
    strokes = []
    canvas.delete("all")
    update_canvas_scrollregion()
    reset_canvas_view()
    current_image_path = image_list[current_index]
    current_latex = latex_list[current_index] if current_index < len(latex_list) else ""
    if not render_image_preview(show_error=True):
        return
    image_name_var.set(f"{os.path.basename(current_image_path)}\nLatex: {current_latex}")
    percent = int(((current_index + 1) / total) * 100)
    progress_var.set(f"{current_index + 1} / {total} ({percent}%)")
    progress_bar["maximum"] = max(total, 1)
    progress_bar["value"] = current_index
    update_status("Draw with the left mouse button. Press Ctrl+S to save quickly.")
    canvas.focus_set()

def render_image_preview(show_error=False):
    global img_tk
    if not current_image_path:
        image_panel.config(image="", text="Select a folder to start")
        return False
    try:
        with Image.open(current_image_path) as img:
            target = int(max(PREVIEW_MIN_SIZE, min(PREVIEW_MAX_SIZE, preview_size)))
            img.thumbnail((target, target), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
        image_panel.config(image=img_tk, text="")
        return True
    except Exception as exc:
        image_panel.config(image="", text="Cannot open image")
        if show_error:
            messagebox.showerror(
                "Cannot open image",
                f"{current_image_path}\n\n{exc}",
                parent=root,
            )
            update_status("Unable to open image. Please check the file and try again.")
        return False

def show_completion():
    global strokes, current_image_path
    strokes = []
    canvas.delete("all")
    update_canvas_scrollregion()
    reset_canvas_view()
    current_image_path = ""
    image_panel.config(image="", text="All images completed!")
    total = len(image_list)
    if total:
        progress_var.set(f"{total} / {total} (100%)")
        progress_bar["maximum"] = total
        progress_bar["value"] = total
    else:
        progress_var.set("0 / 0")
        progress_bar["maximum"] = 1
        progress_bar["value"] = 0
    update_status("All images are complete. Choose another folder or go back with Ctrl+Left.")


def goto_index(index):
    global current_index
    if not image_list:
        return
    total = len(image_list)
    current_index = max(0, min(index, total))
    # Không còn dùng save_progress với current_folder nữa
    if current_index >= total:
        show_completion()
    else:
        show_image()

def start_draw(event):
    global strokes
    if current_index >= len(image_list):
        return
    x = int(canvas.canvasx(event.x))
    y = int(canvas.canvasy(event.y))
    strokes.append(
        {
            "color": pen_color,
            "width": pen_width,
            "points": [(x, y, time.time())],
        }
    )
    canvas.focus_set()

def draw(event):
    if not strokes:
        return
    stroke = strokes[-1]
    x = int(canvas.canvasx(event.x))
    y = int(canvas.canvasy(event.y))
    stroke["points"].append((x, y, time.time()))
    if len(stroke["points"]) >= 2:
        x0, y0, _ = stroke["points"][-2]
        canvas.create_line(
            x0,
            y0,
            x,
            y,
            fill=stroke["color"],
            width=stroke["width"],
            capstyle=tk.ROUND,
            smooth=True,
        )
        update_canvas_scrollregion()

def redraw_canvas():
    canvas.delete("all")
    for stroke in strokes:
        points = stroke["points"]
        for i in range(1, len(points)):
            x0, y0, _ = points[i - 1]
            x1, y1, _ = points[i]
            canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                fill=stroke["color"],
                width=stroke["width"],
                capstyle=tk.ROUND,
                smooth=True,
            )
    update_canvas_scrollregion()

def undo(event=None):
    if not strokes:
        update_status("No strokes to undo.")
        return
    strokes.pop()
    redraw_canvas()
    update_status("Removed the last stroke.")

def clear_canvas(event=None):
    if not strokes:
        canvas.delete("all")
        update_canvas_scrollregion()
        reset_canvas_view()
        return
    if messagebox.askyesno(
        "Clear strokes",
        "Clear all strokes for this image?",
        parent=root,
    ):
        strokes.clear()
        canvas.delete("all")
        update_canvas_scrollregion()
        reset_canvas_view()
        update_status("Cleared all strokes.")

def change_pen_width(value):
    global pen_width
    try:
        pen_width = max(1, int(float(value)))
    except (TypeError, ValueError):
        return
    if pen_value_var is not None:
        pen_value_var.set(f"{pen_width}px")
    update_status(f"Pen width: {pen_width}px")

def set_active_color(color):
    for btn, value in color_buttons:
        btn.configure(relief="sunken" if value == color else "raised")

def change_pen_color(color):
    global pen_color
    pen_color = color
    set_active_color(color)
    update_status(f"Changed pen color to {color}.")

def change_preview_size(value):
    global preview_size
    try:
        preview_size = float(value)
    except (TypeError, ValueError):
        return
    preview_size = max(PREVIEW_MIN_SIZE, min(PREVIEW_MAX_SIZE, preview_size))
    if preview_size_var is not None:
        preview_size_var.set(preview_size)
    if preview_size_value_var is not None:
        preview_size_value_var.set(f"{int(preview_size)}px")
    if image_list and 0 <= current_index < len(image_list):
        render_image_preview()
    update_status(f"Reference size: {int(preview_size)}px")



def save_ink(event=None):
    if not image_list:
        update_status("No generated images. Click 'Generate Images' to start.")
        return
    if current_index >= len(image_list):
        update_status("You have finished all images.")
        return
    if not strokes:
        if messagebox.askyesno(
            "No strokes",
            "No strokes were drawn. Mark this image as done and continue?",
            parent=root,
        ):
            update_status("Image marked as done without strokes.")
            goto_index(current_index + 1)
        else:
            update_status("Continue drawing, then press save.")
        return
    # Đảm bảo output_dir nằm trong folder đã chọn
    try:
        base_name = os.path.splitext(os.path.basename(image_list[current_index]))[0]
        ink = ET.Element("ink")
        trace_group = ET.SubElement(ink, "traceGroup")
        for stroke in strokes:
            points = stroke["points"]
            if len(points) < 2:
                continue
            trace = ET.SubElement(trace_group, "trace")
            coords = " ".join(f"{x},{y}" for x, y, _ in points)
            trace.text = coords
        # Lưu latex vào thuộc tính
        ink.set("latex", current_latex)
        out_file = os.path.join(output_dir, f"{base_name}.ink")
        ET.ElementTree(ink).write(out_file, encoding="utf-8", xml_declaration=True)
        update_status(f"Saved {base_name}.ink to {output_dir}")
    except Exception as exc:
        messagebox.showerror(
            "Save failed",
            f"Could not save the ink file.\n\n{exc}",
            parent=root,
        )
        update_status("Could not save the ink file.")
        return
    goto_index(current_index + 1)

def skip_image(event=None):
    if not image_list:
        return
    if current_index >= len(image_list):
        update_status("You have finished all images.")
        return
    if not confirm_discard():
        return
    update_status("Skipped image without saving.")
    goto_index(current_index + 1)

def previous_image(event=None):
    if not image_list:
        return
    if current_index == 0:
        update_status("Already at the first image.")
        return
    if not confirm_discard():
        return
    goto_index(current_index - 1)

def show_help(event=None):
    help_text = (
        "1. Choose an image folder with the button.\n"
        "2. Draw annotations in the canvas using the left mouse button.\n"
        "3. Use Ctrl+S to save and move to the next image.\n"
        "4. Ctrl+Z undoes the last stroke, Ctrl+Shift+Z clears everything.\n"
        "5. Ctrl+Right skips the image, Ctrl+Left goes back.\n"
        "6. Ink files are written to an 'inks' folder inside the image folder."
    )
    messagebox.showinfo("Quick help", help_text, parent=root)

def on_mousewheel(event):
    if canvas is None:
        return
    delta = event.delta
    if delta == 0:
        return
    step = -1 if delta > 0 else 1
    if event.state & 0x0001:
        canvas.xview_scroll(step, "units")
    else:
        canvas.yview_scroll(step, "units")
    return "break"

def on_mousewheel_linux(event):
    if canvas is None:
        return
    if event.num == 4:
        step = -1
    elif event.num == 5:
        step = 1
    else:
        return
    if event.state & 0x0001:
        canvas.xview_scroll(step, "units")
    else:
        canvas.yview_scroll(step, "units")
    return "break"

def on_exit():
    if confirm_discard():
        root.destroy()

root = tk.Tk()
root.title("Ink Annotation Tool")
try:
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="Segoe UI", size=10)
    root.option_add("*Font", default_font)
except tk.TclError:
    pass
try:
    root.state("zoomed")
except tk.TclError:
    try:
        root.attributes("-zoomed", True)
    except tk.TclError:
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.geometry(f"{screen_w}x{screen_h}")
root.minsize(960, 720)

style = ttk.Style(root)
try:
    if "vista" in style.theme_names():
        style.theme_use("vista")
except tk.TclError:
    pass
style.configure("TButton", padding=(12, 6))
style.configure("TLabelframe", padding=8)

root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

status_var = tk.StringVar(value="Select an image folder to begin.")
progress_var = tk.StringVar(value="0 / 0")
image_name_var = tk.StringVar(value="No image loaded")
folder_var = tk.StringVar(value="Folder: (none)")
pen_value_var = tk.StringVar(value=f"{pen_width}px")
preview_size_var = tk.DoubleVar(value=preview_size)
preview_size_value_var = tk.StringVar(value=f"{int(preview_size)}px")

main = ttk.Frame(root, padding=12)
main.grid(row=0, column=0, sticky="nsew")
main.columnconfigure(0, weight=1)
main.rowconfigure(1, weight=1)

# Top controls
top_bar = ttk.Frame(main)
top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
top_bar.columnconfigure(1, weight=1)




# Đổi nút thành "Generate Images" (tạo trực tiếp sau khi đã định nghĩa auto_generate_images)
btn_folder = ttk.Button(top_bar, text="Generate Images", command=auto_generate_images)
btn_folder.grid(row=0, column=0, padx=(0, 8))

folder_label = ttk.Label(top_bar, textvariable=folder_var, anchor="w")
folder_label.grid(row=0, column=1, sticky="ew")

progress_label = ttk.Label(top_bar, textvariable=progress_var, width=18, anchor="e")
progress_label.grid(row=0, column=2, padx=(12, 0))

progress_bar = ttk.Progressbar(top_bar, length=220, mode="determinate")
progress_bar.grid(row=0, column=3, padx=(6, 0))
progress_bar["maximum"] = 1
progress_bar["value"] = 0

content = ttk.Frame(main)
content.grid(row=1, column=0, sticky="nsew")
content.columnconfigure(1, weight=1)
content.rowconfigure(0, weight=1)

preview_frame = ttk.Frame(content)
preview_frame.grid(row=0, column=0, sticky="n", padx=(0, 16))
preview_frame.columnconfigure(0, weight=1)

preview_title = ttk.Label(preview_frame, text="Reference image", anchor="center")
preview_title.pack(anchor="center")

image_name_label = ttk.Label(
    preview_frame,
    textvariable=image_name_var,
    anchor="center",
    wraplength=320,
    justify="center",
)
image_name_label.pack(anchor="center", pady=(4, 8))

image_panel = ttk.Label(
    preview_frame,
    text="Select a folder to start",
    anchor="center",
    relief="solid",
    padding=20,
)
image_panel.pack()

preview_controls = ttk.Frame(preview_frame)
preview_controls.pack(fill="x", pady=(12, 0))
preview_controls.columnconfigure(1, weight=1)

ttk.Label(preview_controls, text="Reference size:").grid(row=0, column=0, sticky="w")
preview_slider = ttk.Scale(
    preview_controls,
    from_=PREVIEW_MIN_SIZE,
    to=PREVIEW_MAX_SIZE,
    orient="horizontal",
    command=change_preview_size,
    variable=preview_size_var,
)
preview_slider.grid(row=0, column=1, sticky="ew", padx=(6, 6))
ttk.Label(preview_controls, textvariable=preview_size_value_var, width=8).grid(
    row=0, column=2, sticky="e"
)

ttk.Label(
    preview_controls,
    text="Use the slider to zoom the reference image.",
    foreground="#555555",
    wraplength=320,
    justify="left",
).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

canvas_frame = ttk.Frame(content)
canvas_frame.grid(row=0, column=1, sticky="nsew")
canvas_frame.rowconfigure(1, weight=1)
canvas_frame.columnconfigure(0, weight=1)

canvas_title = ttk.Label(canvas_frame, text="Drawing area")
canvas_title.grid(row=0, column=0, sticky="w")

canvas_container = ttk.Frame(canvas_frame)
canvas_container.grid(row=1, column=0, sticky="nsew")
canvas_container.rowconfigure(0, weight=1)
canvas_container.columnconfigure(0, weight=1)

canvas = tk.Canvas(
    canvas_container,
    bg="white",
    width=900,
    height=720,
    highlightthickness=1,
    highlightbackground="#d0d0d0",
    cursor="crosshair",
    xscrollincrement=20,
    yscrollincrement=20,
)
canvas.grid(row=0, column=0, sticky="nsew")

y_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
y_scrollbar.grid(row=0, column=1, sticky="ns")
x_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=canvas.xview)
x_scrollbar.grid(row=1, column=0, sticky="ew")

canvas.configure(
    xscrollcommand=x_scrollbar.set,
    yscrollcommand=y_scrollbar.set,
    scrollregion=(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT),
)

controls = ttk.Frame(main)
controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
controls.columnconfigure(0, weight=1)
controls.columnconfigure(1, weight=0)

buttons_frame = ttk.Frame(controls)
buttons_frame.grid(row=0, column=0, sticky="w")

btn_previous = ttk.Button(buttons_frame, text="< Previous", command=previous_image)
btn_previous.grid(row=0, column=0, padx=4, pady=4)

btn_save = ttk.Button(buttons_frame, text="Save & Next", command=save_ink)
btn_save.grid(row=0, column=1, padx=4, pady=4)

btn_skip = ttk.Button(buttons_frame, text="Skip >", command=skip_image)
btn_skip.grid(row=0, column=2, padx=4, pady=4)

btn_undo = ttk.Button(buttons_frame, text="Undo", command=undo)
btn_undo.grid(row=0, column=3, padx=4, pady=4)

btn_clear = ttk.Button(buttons_frame, text="Clear", command=clear_canvas)
btn_clear.grid(row=0, column=4, padx=4, pady=4)

btn_help = ttk.Button(buttons_frame, text="Help", command=show_help)
btn_help.grid(row=0, column=5, padx=4, pady=4)

pen_frame = ttk.LabelFrame(controls, text="Pen settings")
pen_frame.grid(row=0, column=1, sticky="e", padx=(12, 0))

ttk.Label(pen_frame, text="Width").grid(row=0, column=0, padx=(0, 6), pady=(4, 0))

pen_slider = tk.Scale(
    pen_frame,
    from_=1,
    to=12,
    orient="horizontal",
    showvalue=False,
    length=160,
    command=change_pen_width,
)
pen_slider.set(pen_width)
pen_slider.grid(row=0, column=1, pady=(4, 0))

ttk.Label(pen_frame, textvariable=pen_value_var, width=6).grid(
    row=0, column=2, padx=(6, 0), pady=(4, 0)
)

ttk.Label(pen_frame, text="Color").grid(row=1, column=0, sticky="w", pady=(6, 2))

colors = ["black", "red", "blue", "green", "orange", "purple"]
for idx, color in enumerate(colors):
    btn = tk.Button(
        pen_frame,
        bg=color,
        width=3,
        relief="raised",
        command=lambda c=color: change_pen_color(c),
    )
    btn.grid(row=1, column=1 + idx, padx=2, pady=(4, 6))
    color_buttons.append((btn, color))

instruction_label = ttk.Label(
    main,
    text=INSTRUCTION_TEXT,
    justify="left",
    foreground="#555555",
    wraplength=900,
)
instruction_label.grid(row=3, column=0, sticky="ew", pady=(12, 0))

status_bar = tk.Label(
    root,
    textvariable=status_var,
    anchor="w",
    relief="sunken",
    borderwidth=1,
    padx=8,
    pady=4,
)
status_bar.grid(row=1, column=0, sticky="ew")

set_active_color(pen_color)
update_canvas_scrollregion()
reset_canvas_view()

canvas.bind("<ButtonPress-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<MouseWheel>", on_mousewheel)
canvas.bind("<Shift-MouseWheel>", on_mousewheel)
canvas.bind("<Button-4>", on_mousewheel_linux)
canvas.bind("<Button-5>", on_mousewheel_linux)
canvas.bind("<Shift-Button-4>", on_mousewheel_linux)
canvas.bind("<Shift-Button-5>", on_mousewheel_linux)

root.bind_all("<Control-s>", save_ink)
root.bind_all("<Control-z>", undo)
root.bind_all("<Control-Shift-Z>", clear_canvas)
root.bind_all("<Control-Right>", skip_image)
root.bind_all("<Control-Left>", previous_image)
root.bind_all("<F1>", show_help)

root.protocol("WM_DELETE_WINDOW", on_exit)

update_status("Select an image folder to begin.")
root.mainloop()

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from regress_trees.cart import *
import tkinter as tkinter


def draw_new_tree():
    tol_n, tol_s = get_inputs()
    re_draw(tol_s, tol_n)


def re_draw(tol_s, tol_n):
    # 清空画布内容
    f.clf()
    # 获得一个子图
    axis = f.add_subplot(111)

    if chbBtnVar.get():
        if tol_n < 2:
            tol_n = 2
        my_tree = create_tree(raw_data, model_leaf, model_err, (tol_s, tol_n))
        y_hat = create_fore_cast(my_tree, test_data, model_tree_eval)
    else:
        my_tree = create_tree(raw_data, ops=[tol_s, tol_n])
        y_hat = create_fore_cast(my_tree, test_data)

    x0 = raw_data[:, 0].T.tolist()[0]
    x1 = raw_data[:, 1].T.tolist()[0]
    # axis.scatter(raw_data[:, 0], raw_data[:, 1], s=5)
    axis.scatter(x0, x1, s=5)
    axis.plot(test_data, y_hat, linewidth=2.0)
    canvas.show()


def get_inputs():
    try:
        tol_n = int(tolNentry.get())
    except BaseException as e1:
        tol_n = 10
        print('Enter integer for tol_n')
        tolNentry.delete(0, tkinter.END)
        tolNentry.insert(0, '10')

    try:
        tol_s = float(tolSentry.get())
    except BaseException as e2:
        tol_s = 1.0
        print('Enter float for tols')
        tolSentry.delete(0, tkinter.END)
        tolSentry.insert(0, '1.0')

    return tol_n, tol_s


root = tkinter.Tk()

# Label 控件: 显示文本信息
# grid() 方法设定控件的位置, 位于哪一行,那一列
# 指定 columnspan, rowspan 的值允许一个小部件跨行或者跨列
# tkinter.Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)

f = Figure(figsize=(5, 4), dpi=100)
# 指定master=root, 表示画布放入到控件中
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
# 将画布放置到控件上,第0行,占3列
canvas.get_tk_widget().grid(row=0, columnspan=3)

tkinter.Label(root, text='tolN').grid(row=1, column=0)

# Entry 部件 : 文本输入框, 允许单行文本输入的文本框
tolNentry = tkinter.Entry(root)
tolNentry.grid(row=1, column=1)
# 设置默认输入值
tolNentry.insert(0, '10')

tkinter.Label(root, text='tolS').grid(row=2, column=0)

tolSentry = tkinter.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

# Button: 点击按钮
tkinter.Button(root, text='ReDraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

# IntVar: 按钮整数值
# 用于Checkbutton中选项选择
chbBtnVar = tkinter.IntVar()

# Checkbutton: 复选框
chkBtn = tkinter.Checkbutton(root, text='Model Tree', variable=chbBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

raw_data = np.mat(load_data_set('sine.txt'))
raw_data0 = raw_data[:, 0].T.tolist()[0]
test_data = np.arange(min(raw_data0), max(raw_data0), 0.01)

re_draw(1.0, 10)
root.mainloop()

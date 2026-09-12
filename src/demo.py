from genericpath import exists
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f

from pandas.core.reshape import encoding
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd 

ROOT = Path(r"C:\Users\Admin\Desktop\Demo_python\demo\demo")
file_name = (r"C:\Users\Admin\Desktop\Demo_python\demo\demo\DATA.xlsx")   #đọc tên file
df = pd.read_excel(file_name, sheet_name="DATA GỐC")
df.columns = df.columns.str.strip()  # loại bỏ khoảng trắng 
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

varr_x1 =[
    'Lang nghe trong lop hoc',
    'Thao luan giup cai thien hung thu',
    'Lop hoc dao nguoc'
    ]
df['Chất Lượng Giảng Dạy'] = df[varr_x1].mean(axis=1) #chia theo hàng ngang 

varr_x2 =[
    'Loai hinh cho o tai Sip',
    'Phuong tien di lai den truong dai hoc'
    ]
df['CSVC'] = df[varr_x2].mean(axis=1)

varr_x3 =[
    'Tan suat doc sach phi khoa hoc',
    'Tan suat doc sach khoa hoc',
    'Ghi chep trong lop hoc',
    'Chuan bi ky thi giua ky 1',
    'Chuan bi ky thi giua ky 2'
]
df['Học Liệu Số'] = df[varr_x3].mean(axis=1)

varr_x4 =[
    'Cong viec lam them',
    'Tac dong cua du an/hoat dong'
    ]
df['Việc Làm'] = df[varr_x4].mean(axis=1)

varr_x5 =[
    'Tham du hoi thao chuyen nganh',
    'Tham du cac lop hoc'
    ]
df['Ngoại Khóa'] = df[varr_x5].mean(axis=1)


#Phân lối khối ngành   #def dùng để định nghĩa hàm 
def Khoi_Nganh(ma_so_mon_hoc):
    if 1 <= ma_so_mon_hoc <= 4:
        return "CNTT - Kỹ Thuật"
    elif 5 <= ma_so_mon_hoc <= 9:
         return "Quản Trị - Kinh Tế"
    else:
        return "Khác"
df["Khối Ngành"] = df['Ma so mon hoc'].apply(Khoi_Nganh)       

#Tính ước lượng MLE vector trung bình của sinh viên
varr_x6 = [
   'Chất Lượng Giảng Dạy',
   'CSVC',
   'Học Liệu Số',
   'Việc Làm',
   'Ngoại Khóa'
    ]
#gourpby nhóm dữ liệu (Khối Ngành) theo cột phân loại
mle_vtr =(df.groupby("Khối Ngành")[varr_x6].mean().round(2))
print("\n--- Ước Lượng MLE vector Trung Bình Của Từng Khối Ngành ---")
print(mle_vtr)

#Kiểm định Hotelling’s T2          
#1. khối CNTT - Kỹ Thuật
#2. khối Quản Trị - Kinh Tế
#Tách dữ liệu từng khối ngành thành ma trận
#values chuyển dataframe thành ma trận 
nhom_cntt = df[df["Khối Ngành"] == "CNTT - Kỹ Thuật"][varr_x6].values
nhom_quantri = df[df["Khối Ngành"] == "Quản Trị - Kinh Tế"][varr_x6].values

#.shape trả về dạng cặp (số dòng , số cột)
# n1, p là số dòng + số biến của cntt
# n2, _ là số dòng của khoa quantri và _ có nghĩa là p của quantri = p cntt (=5)
n1, p = nhom_cntt.shape
n2, _ = nhom_quantri.shape

#Tính vector trung bình mẫu của hai khối ngành (x1,x2)
#mean giá trị trung bình 
#axis = 0 chia theo hàng dọc
mean1 = np.mean(nhom_cntt, axis=0)
mean2 = np.mean(nhom_quantri, axis = 0)

#Tính ma trận hiệp phương sai S_p
#rowvar = False: mỗi cột là p biến, mỗi dòng là n quan sát
#rowvar = True : mỗi cột là n quan sát , mỗi dòng là p biến
#N - g = n1 + n2 - 2 
s1 = np.cov(nhom_cntt, rowvar = False)
s2 = np.cov(nhom_quantri, rowvar = False)
S_p = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)

#Tính khoảng cách Hotelling T2
#diff là sự chênh lệch giữa cntt (mean1) và quantri(mean2)
#np.linalg.inv(S_p) tìm ma trận khả nghịch của S_P
#diff.T là chuyển vị 
# @ là phép nhân hai ma trận

diff = mean1 - mean2
T2 = ((n1 * n2) / (n1 + n2)) * diff.T @ np.linalg.inv(S_p) @ diff

# Chuyển đổi T2 sang F và giá trị p-value
#f.cdf tinh xac suat tu diem 0 di len
F_stat = ((n1 + n2 - p - 1) / (n1 + n2 - 2) * p) *T2
p_value = 1 - f.cdf(F_stat, p, n1 + n2 - p - 1)

#In Kết Quả
print("\n ---Kiểm Định Hotelling T2 Cho 2 Khối Ngành---")
print(f"Thống kê T2:  {T2:.4f}")
print(f"Giá trị F  :  {F_stat:.4f}")
print(f"p-value    :  {p_value:.4f}")

if p_value < 0.005:
    print("\nCó sự khác biệt thống kê giữa vector trung bình của  2 khối ngành\n")
else:
    print("\nKhông đủ bằng chứng để bác bỏ giải thuyết vector trung bình của 2 khối ngành giống nhau\n")



# Chỉ hiển thị các cột cần xem 
cot_hien_thi = (
    ['STUDENT_ID'] 
    + ['Khối Ngành']
    + ['Chất Lượng Giảng Dạy'] 
    + ['CSVC'] 
    + ['Học Liệu Số']
    + ['Việc Làm']
    + ['Ngoại Khóa']
)

print(df[cot_hien_thi].round(2).head(3)) #round(2) làm tròn 2 số sau dấu phẩy
print(df[cot_hien_thi].round(2).tail(3))

#3. Gọi hàm vẽ biểu đồ bằng danh sách 5 tiêu chí điểm số này:
#3.1. Vẽ biểu đồ thể hiện trung bình 
mle_T = mle_vtr.T #chuyển vị về thành cột 
labels ={"x1":"Giảng Dạy", "x2":"CSVC",
         "x3":"Học Liệu Số","x4":"Hỗ Trợ Việc Làm"
         ,"x5":"Hoạt Động Ngoại Khóa"}
khoi_nganh = ["CNTT - Kỹ Thuật", "Quản Trị - Kinh Tế"]
varr_x7 = [
   'Chất Lượng Giảng Dạy',
   'CSVC',
   'Học Liệu Số',
   'Việc Làm',
   'Ngoại Khóa'
    ]
#fig toàn bộ hình
#ax khu vực dùng để vẽ biểu đồ
#figsize=(7.2, 4.5) kích thước biểu đồ
fig, ax = plt.subplots(figsize = (7.2, 4.5))

# Duyệt qua từng khối ngành trong danh sách để vẽ
for kn in khoi_nganh:
    # varr_x7: Trục x là danh sách các tiêu chí
    # mle_T[kn].to_numpy(): Trục y là giá trị trung bình MLE của hai khối ngành
    # label = kn: Đặt tên nhãn cho đường để hiển thị trong bảng chú thích
    ax.plot(varr_x7,
            mle_T[kn].to_numpy(), label = kn)
ax.set_ylabel("Dữ liệu đánh giá") #Đặt tên trục y
ax.set_xlabel("Tên biến") #Đặt tên trục X
ax.set_title("Thông tin trung bình trong khối ngành") #Đặt tên tiêu đề         
ax.legend() #Hiển thị chú thích
fig.tight_layout() #Căn chỉnh biểu đồ
fig.savefig(FIG / "chart_demo.png", dpi =180) # Lưu biểu đồ thành file ảnh PNG với độ phân phải là 180
plt.close (fig) # Đóng biểu đổ

#3.2. Vẽ biểu đồ thể hiện sự phân cụm 
for x in varr_x7: #Lần lượt duyệt qua từng biến trong varr_x7
    fig, ax = plt.subplots(figsize=(6.4, 4.4)) #Tạo khung vẽ với kích thước là 6.4, 4.4
    
    # Lọc data của từng khối ngành trong danh sách, lấy ra tên khối ngành để đặt cho trục X
    # rồi chuyển về dạng Numpy để vẽ các điểm dữ liệu 
    arrays=[df.loc[df['Khối Ngành'].eq(kn),x].to_numpy() for kn in khoi_nganh]

    # Vẽ biểu đồ boxplot cho dữ liệu đã lọc, đặt tên cho trục X
    ax.boxplot(arrays, tick_labels = khoi_nganh)

    #Vòng lặp duyệt qua từng mảng dữ liệu nhóm ngành để vẽ các điểm dữ liệu 
    for j, vals in enumerate(arrays, start = 1):

        #Tạo khoảng chêch lệch (jitter) với -0.08 và 0.08
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), j) + jitter, vals)

    ax.set_title(f"{x}") # Đặt tiêu đề cho biểu đồ dựa theo tên biến hiện tại (x)
    ax.set_ylabel("Thông số") # Đặt tên trục y
    ax.set_xlabel("Tên biến") # Đặt tên trục X
    fig.tight_layout() #Căn chỉnh biểu đồ
    fig.savefig(FIG / f"demo_{x}.png", dpi = 180) # Lưu biểu đồ thành file ảnh PNG với độ phân phải là 180
    plt.close(fig) # Đóng biểu đổ


# Hàm xuất Data ra file CSV
# Khai bán hàm để xuất data ra file csv
def export_file_csv(data_frame, file_path="demo.csv", columns=None):
    try: #Kiểm tra lỗi nếu có thì chuyển xuống except
        if columns is not None: #Nếu có, lấy các cột nằm trong columns
            export_df = data_frame[columns]

        else: #Nếu không lấy toàn bộ cột 
            export_df = data_frame 

        export_df = export_df.round(2) #Làm tròn các cột 2 số sau dấu phẩy 

        # index=False: không xuất số thứ tự
        # encoding ="utf-8-sig": hiển thị tiếng Việt 
        # sep=",": ngăn cách các cột bằng dấu ,
        export_df.to_csv(file_path, index = False, encoding ="utf-8-sig", sep = ",")
        print(f"Đã xuất thành công file CSV {file_path}")

    # Hiển thị lỗi nếu file xuất không được 
    except Exception as e:
        print(f"Xuất file không thành công: {e}")

# Gọi hàm để lấy các cột cần xem 
export_file_csv(df,file_path="Báo Cáo Thống Kê.csv", columns = cot_hien_thi)
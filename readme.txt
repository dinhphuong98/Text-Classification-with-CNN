Text Classification with CNN
Mô tả:
Bài toán phân loại các bài báo dựa trên title của chúng.
với 10 lớp khác nhau
Các bước thực hiện:
step 1: tách từ ( tokenize )
- sử dụng thư viện pyvi.ViTokenizer
step 2: loại bỏ từ dừng (stop word ), ký tự đặc biệt, số
- từ dừng là những từ ít quan trọng và không mang nhiều ý nghĩa như: " thì, là, ở, ...
- các ký tự đặc biệt và số #,%,^,1,2
step 3: 
- word embedding: được train trên tập wiki tiếng việt khoảng 10000 từ.
- mỗi word sẽ được đại diện bở vector 100 chiều
- lấy 15 word đầu tiên của title - > matrix đầu vào kích thức 15*100
step 4: training
- fit vào mạng CNN
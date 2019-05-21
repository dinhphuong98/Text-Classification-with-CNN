
import torch.nn.functional as F
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchsummary import summary
import pickle
##-------------------------------------##

class Net(nn.Module):
    
    def __init__(self,num_class):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.mp = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1472, 640)
        torch.nn.init.xavier_uniform(self.fc.weight)
        self.fc1 = nn.Linear(640, num_class)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc1(x)
       
        return F.log_softmax(x)

def convert_out_to_class(out):
    '''
        out is a pytorch tensor
    '''
    out = out.detach()[0].tolist()
    out_max = max(out)
    return out.index(out_max)

from gensim.models import Word2Vec
import os
from pyvi import ViTokenizer
import torch
import pickle
from torch.autograd import Variable

##--------------------------------------------##

stop_words =[ 'bị',
'bởi',
'cả',
'các',
'cái',
'cần',
'càng',
'chỉ',
'chiếc',
'cho',
'chứ',
'chưa',
'chuyện',
'có',
'có_thể',
'cứ',
'của',
'cùng',
'cũng',
'đã',
'đang',
'đây',
'để',
'đến_nỗi',
'đều',
'điều',
'do',
'đó',
'được',
'dưới',
'gì',
'khi',
'không',
'là',
'lại',
'lên',
'lúc',
'mà',
'mỗi',
'một_cách',
'này',
'nên',
'nếu',
'ngay',
'nhiều',
'như',
'nhưng',
'những',
'nơi',
'nữa',
'phải',
'qua',
'ra'
'rằng',
'rằng',
'rất',
'rất',
'rồi',
'sau',
'sẽ',
'so',
'sự',
'tại',
'theo',
'thì',
'trên',
'trước',
'từ',
'từng',
'và',
'vẫn',
'vào',
'vậy',
'vì',
'việc',
'1',
'0',
'2',
'3',
'4',
'5',
'6',
'7',
'8',
'9',
'với',
'vừa',
'.',',',
'/','\\','%','*','$','~','`','!','#','^','&','(',')','_','+','-','=',';',':','"',"'",'{','}','[',']'
]

def delete_stop_words(sentense, stop_word = stop_words):
    for index,ele in enumerate(sentense):
        if ele in stop_word:
            del sentense[index]


def get_tokenizer(sentense):
    '''
        read the text then tokenizer
    '''
    
    
    sentense = sentense.lower()
    sentense = ViTokenizer.tokenize(sentense)

    temp = sentense.strip().split()
    delete_stop_words(temp)
    
    return temp


def get_word_embedding(pretrained,sentense = None):
    with open(pretrained, 'rb') as handle:
        vocab_data = pickle.load(handle)
    datasets = []
    temp = [i*0 for i in range(100)]
    lenSen = len(sentense)
    for i in range(15):
        if i < lenSen:
            if sentense[i] in vocab_data:
                datasets.append(vocab_data[sentense[i]])
            else:
                datasets.append(temp)
        else:
            datasets.append(temp)
        
    return datasets



if __name__ == '__main__':
    pretrained = 'vocal.pickle'
    
    title = 'Ro “vẩu” đi làm đẹp Theo báo chí TBN cho hay, thời còn thi đấu cho Barcelona, Ronaldinho giấu rất kín chuyện cắt tóc của mình. Ro “vẩu” chưa bao giờ để các phóng viên tìm được nơi anh chăm sóc mái tóc dài của mình. Tuy nhiên lần này, nhờ sự giới thiệu của các đồng đội, Ronaldinho đã tìm đến thợ làm tóc hàng đầu Milan, Manolo Garcia, và đương nhiên anh không thể thoát khỏi ống kính của các tay săn ảnh. Theo nguồn tin được Marca tiết lộ, Ronaldinho đã giành cả buổi chiều để chỉnh trang lại mái tóc của mình, trước khi dùng bữa tối với mẹ và cô bạn gái mới, Alina Domingos. Tuy nhiên theo tờ The Spoiler của Anh cho hay, R80 muốn sửa lại mái tóc của mình để trình làng diện mạo thật mới trong trận derby Milan – trận đấu anh rất có thể được đá chính do Kaka đang chấn thương. Sau vài giờ đồng hồ ngồi lỳ trên ghế, kiểu đầu mới của Ronaldinho đã hoàn tất. Tuy nhiên vẫn với thói quen cũ, Ro “vẩu” không để các phóng viên chụp được kiểu đầu mới của mình, khi chủ động đội chiếc mũ len trên đầu. Sau khi ký tặng vài CĐV, anh nhanh chóng lên xe đến điểm hẹn. Trang Anh '    
    senten = get_tokenizer(sentense = title)
    
    embedding = get_word_embedding(pretrained=pretrained,sentense=senten)
    x = torch.Tensor([embedding]).view(-1,1,15,100).cuda()
    
    num_class = 10
    filepath = 'pretrained'
   
    model = Net(num_class = num_class)
    model = model.cuda()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    
    out = model(x)
    index = convert_out_to_class(out)
    print("Index: {}".format(index))
    with open('label.pkl', 'rb') as f:
         label = pickle.load(f)
    print('Label: {}'.format(label[index]))
    
    
    
    


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


def get_tokenizer(link):
    '''
        read the text then tokenizer
    '''
    
    with open(link,'r',encoding='utf-8') as f:
        sentense = f.read()
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


def get_embedding_and_label(root_path):
    
    pretrained = 'vocal.pickle'
    print('Building train data...')

    with open('label.pkl', 'rb') as handle:
        label = pickle.load(handle)
    X_data = []
    Y_data = []
    

    for index, folder in enumerate(os.listdir(root_path)):
        label[index] = folder
        for file in os.listdir(os.path.join(root_path,folder)):
            link = os.path.join(root_path,folder,file)
            sentense = get_tokenizer(link=link)

            if sentense != None:
                embedding = get_word_embedding(pretrained=pretrained,sentense=sentense)
            else:
                continue

            x = torch.Tensor([embedding])
            y = Variable(torch.LongTensor([index]),requires_grad = False)
            
            X_data.append(x)
            Y_data.append(y)
           

    return X_data, Y_data,label

def get_data(root_path):
    print('building...')
    X_data = []
    Y_data = []
    pretrained = 'vocal.pickle'
    with open('label.pkl', 'rb') as f:
        label = pickle.load(f)
    for index in label:
        folder = label[index]
        for file in os.listdir(os.path.join(root_path,folder)):
                link = os.path.join(root_path,folder,file)
                sentense = get_tokenizer(link=link)

                if sentense != None:
                    embedding = get_word_embedding(pretrained=pretrained,sentense=sentense)
                else:
                    continue

                x = torch.Tensor([embedding])
                y = Variable(torch.LongTensor([index]),requires_grad = False)
                
                X_data.append(x)
                Y_data.append(y)

    return X_data,Y_data

if __name__ == '__main__':
    
    file = 'train_dataset.pkl'
    with open(file, 'wb') as f:
        data = pickle.load(f)
    
   


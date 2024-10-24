
bword_data = pd.read_table('./fword_list.txt', names=['word'])  
gword_data = pd.read_csv('https://raw.githubusercontent.com/3beol/3beol.github.io/refs/heads/master/%ED%95%9C%EA%B5%AD%EC%96%B4%2B%ED%95%99%EC%8A%B5%EC%9A%A9%2B%EC%96%B4%ED%9C%98%2B%EB%AA%A9%EB%A1%9D.csv')

pre = Preprocessing(bword_data, gword_data)
train_texts, test_texts, train_labels, test_labels, max_length = pre.make_data()

train_batch_size = 16 
eval_batch_size = 16
epochs = 50
weight_decay = 0.001

save_path = './results/fine_tuned_model'

tr = Train(save_path, train_batch_size, eval_batch_size, epochs, weight_decay)
train_model = tr.train_model(train_texts, train_labels, test_texts, test_labels)

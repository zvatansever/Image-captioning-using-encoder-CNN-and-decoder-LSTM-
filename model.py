import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.vocab_size = vocab_size
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(0, 0)
        self.linear.weight.data.uniform_(0, 0)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        out, hiddens = self.lstm(embeddings)
        outputs = self.linear(out[:,:-1,:])
        return outputs
    def sample(self, features, states=None):
        sampled_ids = []
        for i in range(20):                                    # maximum sampling length
            hiddens, states = self.lstm(features, states)       
            outputs = self.linear(hiddens.squeeze(1))      
            predicted = outputs.max(1)[1]
            sampled_ids.append((int)(predicted.cpu().numpy()))
            features = self.embed(predicted)
            features = features.unsqueeze(1)                       
        return sampled_ids